from LogMatcher import LogMatcher
from LogProcessor import postprocess_template, contain_wildcard_only
from SLGSelector import SLGSelector
from LLMQuery import LLMQueryer
from LogEvaluator import LogEvaluator
from settings import datasets
import utils
import time
import os
import pandas as pd
import re
import argparse


def check_again(log, match_result, matcher: LogMatcher):
    if not match_result[0]:
        return (False, None)
    
    template = matcher.templates[match_result[1]]
    template = template.replace('<*>', '')
    chars = list(set(re.findall(r'[^a-zA-Z0-9\s]', template)))
        
    if len(chars) == 0:
        return (True, None)
    
    for char in chars:
        if char not in log:
            match_again_result = matcher.match_template(log, [match_result[1]])
            return (True, match_again_result[1]) if match_again_result[0] else (False, match_result[1])
    return (True, None)
    


models = ['deepseek-chat', 'qwen-plus', 'gpt-4o-2024-08-06', 'claude-3-5-Sonnets']

input_path = "full_dataset"

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Set the model to use', default='gpt-4o-2024-08-06')
parser.add_argument('--candidate', type=str, help='Set the candidate set construction method', default='hierarchical',choices=['hierarchical', 'random'])
parser.add_argument('--similarity', type=str, help='Set sampling similarity algorithm', default="enhanced", choices=['enhanced', 'jaccard', 'cosine'])
parser.add_argument('--sample', type=int, help='Set the number of samples', default=3)
parser.add_argument('--knowledgebase', type=bool, help='Set whether to use the knowledge base', default=True)
parser.add_argument('--strategy', type=int, help='Setting framework strategy', default=2, choices=[1, 2])
args = parser.parse_args()


model = args.model
candidate = args.candidate
similarity_algorithm = args.similarity
sample = args.sample
knowledgebase = args.knowledgebase
strategy = args.strategy



if __name__ == '__main__':

    for dataset in datasets:
        print(f"Processing dataset [{dataset}]:")
        input_file = os.path.join(input_path, dataset, f"{dataset}_full.log_structured.csv")
        input_log = pd.read_csv(input_file)
        file_length = len(input_log)
        log_messages = []
        log_templates = []
        template = None
        matcher = LogMatcher()
        queryer = LLMQueryer(dataset, model)
        selector = SLGSelector(dataset, candidate, similarity_algorithm)
        query_times = 0
        parsing_start_time = time.time()
        for idx, row in enumerate(input_log.itertuples()):
            # step-1
            template_id = -1
            filter_ids = []
            parse_log = str(row.Content)
            match_result = matcher.match_template(parse_log, filter_ids)
            check_result = check_again(parse_log, match_result, matcher)
            if check_result[0]:
                template_id = check_result[1] if check_result[1] is not None else match_result[1]
            else:
                if check_result[1] is not None:
                    filter_ids.append(check_result[1])
                # step-2
                print(f"\n{idx+1}/{file_length} log match failed.")
                
                start_time = time.time()
                # step-2.1
                # select examples from candidate set
                examples = selector.select_examples(parse_log, sample)
                print(f"Sampling took {time.time()-start_time} seconds")

                query_result = queryer.query_template_from_llm(parse_log, 
                                                               examples=examples, 
                                                               model=model,
                                                               knowledgebase=knowledgebase,
                                                               strategy=strategy)
                query_times += 1
                print(f"The query result is: {query_result}")
                template = queryer.extract_template_from_llm(query_result)
                template_extracted = template
                if template is not None:
                    template = postprocess_template(template)
                    print(f"The final template is: `{template}`")
                    # step-3
                    template_id = matcher.insert(parse_log, template)
                    # step-4
                    new_match_result = matcher.match_template(parse_log, filter_ids)
                    if new_match_result[0]:
                        if template_id == new_match_result[1]:
                            template_id = new_match_result[1]
                        else:
                            filter_ids.append(new_match_result[1])
                    else:
                        print("Template is error, re-querying template from LLM.")
                        template, template_id, retry_times = queryer.requery_template_from_llm_with_check(parse_log, 
                                                                                                          template_id, 
                                                                                                          matcher, 
                                                                                                          examples, 
                                                                                                          model, 
                                                                                                          filter_ids,
                                                                                                          knowledgebase,
                                                                                                          strategy)
                        query_times += retry_times

                # step-5
                if template is None:
                    print("No template found, re-querying template from LLM.")
                    template, template_id, retry_times = queryer.requery_template_from_llm(parse_log, 
                                                                                           matcher, 
                                                                                           examples, 
                                                                                           model, 
                                                                                           filter_ids,
                                                                                           knowledgebase,
                                                                                           strategy)
                    query_times += retry_times
                
                # step-6 Check if template is overmatched
                if contain_wildcard_only(template):
                    print(f"Template `{template}` overmatching, re-querying template from LLM.")
                    template_id, retry_times = queryer.requery_template_from_llm_with_erroTemplate(parse_log, 
                                                                                                   template_id, 
                                                                                                   matcher, 
                                                                                                   template_extracted, 
                                                                                                   model, 
                                                                                                   filter_ids,
                                                                                                   knowledgebase,
                                                                                                   strategy)
                    query_times += retry_times
            
            log_messages.append((idx, parse_log))
            log_templates.append((idx, template_id))

        parsing_time = time.time() - parsing_start_time
        print(f"Parsing took: {parsing_time} s")
        print(f"{dataset} has queried {query_times} times.")
        if not os.path.exists(f"result/{dataset}"):
            os.makedirs(f"result/{dataset}")
        
        print(f"file {dataset} is writing...")
        utils.cache_to_file(log_messages, f"result/{dataset}/cached_log_final.txt")
        utils.cache_to_file(log_templates, f"result/{dataset}/cached_template_final.txt")
        pd.to_pickle(matcher, f"result/{dataset}/matcher_final.pkl")
        
        output_file = f"log_structured_{candidate}_{similarity_algorithm}_{sample}_{knowledgebase}_{strategy}_{model}.csv"
        utils.result_to_csv(log_messages, 
                            log_templates, 
                            matcher, 
                            f"result/{dataset}/{output_file}")
        print("file writing done.")


        print(f"evaluating {dataset}...")
        evaluator = LogEvaluator(dataset)
        parsedresult_df = pd.read_csv(f"result/{dataset}/{output_file}")
        groundtruth_df = pd.read_csv(f"full_dataset/{dataset}/{dataset}_full.log_structured_new.csv")
        PA, GA, FGA, FTA, evalute_time = evaluator.evaluate(groundtruth_df, parsedresult_df)
        utils.evaluate_result_to_csv(
            dataset,
            parsing_time,
            evalute_time,
            PA,
            GA,
            FGA,
            FTA,
            candidate,
            similarity_algorithm,
            sample,
            knowledgebase,
            strategy,
            model
        )
        print(f"{dataset} evaluating done.")
    
    print("All datasets are parserd done.")