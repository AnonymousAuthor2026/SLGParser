from openai import OpenAI
import openai
import re
from LogMatcher import LogMatcher
from LogProcessor import postprocess_template, contain_wildcard_only, postprocess_error_template
from typing import Tuple
import json

class LLMQueryer:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.vocab_file = open(f"vocab/{self.dataset}",'r').readlines()
        self.api_key, self.base_url = self.set_model_config(model)

    def set_model_config(self, model):
        with open("api_config.json", "r", encoding='utf-8') as f:
            config = json.load(f)

        if 'gpt' in model:
            api_key = config["gpt"]["key"]
            base_url = config["gpt"]["url"]
        elif "qwen" in model:
            api_key = config["qwen"]["key"]
            base_url = config["qwen"]["url"]
        elif "deepseek" in model:
            api_key = config["deepseek"]["key"]
            base_url = config["deepseek"]["url"]
        elif "claude" in model:
            api_key = config["claude"]["key"]
            base_url = config["claude"]["url"]

        return api_key, base_url

    def select_suitable_prompt(self, log, examples, knowledgebase, strategy):
        messages = []

        if len(examples) == 0 and strategy == 2:
            messages = [
                {'role': 'system', 
                'content': 'You are a log parsing expert. I will provide a log message in backticks. '
                            'Identify dynamic variables and replace them with {placeholder}. '
                            'Output the static log template without extra explanation and keep original format. '
                            'Make full use of semantics to make judgments.'},

                {'role': 'user',
                'content': 'Here are specific rules for parsing:\n'
                            '- Number(s), hexadecimal, dates, and interface are likely dynamic.\n'
                            '- Exception type, error messages, interrupted logs, and methods name are NOT dynamic.\n'
                            '- Path, url, date time are treated as ONE single variable.\n'},
                
                {'role': 'user',
                'content': 'Log message: `Code:0,1 Receiving block blk_-114514 src: /10.250.19.102:50010 dest: /10.250.19.102:443`\n'},
                
                {'role': 'assistant',
                'content': 'Log template: `Code:{code1},{code2} Receiving block {block_id} src: {src_ip}:{port} dest: {dest_ip}:{port}`'},
                        
                {'role': 'user', 
                'content': 'The following is the log to be parsed.\n'
                            f'Log message: `{log}`'},
            ]
            if knowledgebase:
                vocab_set = set()
                for token in re.split(r'[:\s.()./#,\[\]!?;]+', log):
                    token = token.rstrip(':.=,!;([{"\'+')
                    token = token.lstrip(',=:)]}"\'')
                    if token in self.vocab_file:
                        vocab_set.add(token.strip())
                vocab = " ".join(vocab_set)
                messages.append(
                    {'role': 'user',
                     'content': 'Here are some words that may be part of static template in Log message:\n'
                                f'{vocab}'}
                )
    
        else:
            
            messages = [
                {'role': 'system', 
                'content': 'You are a log parsing expert. I will provide a log message in backticks. '
                            'Identify dynamic variables and replace them with {placeholder}. '
                            'Output the static log template without extra explanation and keep original format.'},

                {'role': 'user',
                'content': 'Here are specific rules for parsing:\n'
                            '- Number(s), hexadecimal, dates, and interface are likely dynamic.\n'
                            '- Exception type, error messages, interrupted logs, and methods name are NOT dynamic.\n'
                            '- Path, url, date time, digit-unit are treated as ONE single variable.\n'},
                
                {'role': 'user',
                'content': 'Log message: `Code:0,1 Receiving block blk_-114514 src: /10.250.19.102:50010 dest: /10.250.19.102:443`\n'},
                
                {'role': 'assistant',
                'content': 'Log template: `Code:{code1},{code2} Receiving block {block_id} src: {src_ip}:{port} dest: {dest_ip}:{port}`'},
            ]
            if not len(examples) == 0:
                example_text = '\n'.join([f"`{example}`" for example in examples])
                messages.append(
                    {'role': 'user', 
                     'content': 'Here are some logs similar to Log message:\n'
                                f'Similar logs: {example_text}\n'
                                'Log message and similar logs are likely to share the same template.'}
                )
            messages.append(
                {'role': 'user', 
                 'content': 'The following is the log to be parsed.\n'
                            f'Log message: `{log}`'},
            )

        return messages
    

    def query_template_from_llm(self, 
                                log: str, 
                                examples: list = [], 
                                errorMessage: str = None, 
                                model: str = 'gpt-4o-mini',
                                knowledgebase: bool = True,
                                strategy: int = 2):
        """
        Query template from LLM according to the raw log.
        returns:
            response: The query result that LLM responses.
        """
        retry_times = 0
        while retry_times < 3:
            try:
                messages = self.select_suitable_prompt(log, examples, knowledgebase, strategy)
                client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                if errorMessage is not None:
                    messages.append({'role': 'user', 
                                    'content': f'The template parsed earlier has excessive matching or error: `{errorMessage}`. '
                                    'Please parse it again!'})

                completion = client.chat.completions.create(
                    model = model,
                    messages = messages,
                    temperature = 0.0,
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"Error! Error message: {e}")
            retry_times += 1
        print(f"Failed to get response from LLM after {retry_times} retries.")
        return None


    def extract_template_from_llm(self, response):
        """
        Extract the template from the LLM response.

        returns:
            template: if extract template successfully, then return `template` , othewise return None.
        """
        if response is None:
            return None
        lines = response.split(r'\n')
        
        log_template = None
        for line in lines:
            if line.find("Log template:") != -1:
                log_template = line
                break
        if log_template is None:
            for line in lines:
                if line.find("`") != -1:
                    log_template = line
                    break
        if log_template is not None:
            start_index = log_template.find('`') + 1
            end_index = log_template.rfind('`')

            if start_index == 0 or end_index == -1:
                start_index = log_template.find('"') + 1
                end_index = log_template.rfind('"')

            if start_index != 0 and end_index != -1 and start_index < end_index:
                template = log_template[start_index:end_index]
                template = re.sub(r'`','',template)
                return template
        
        if len(lines) == 1:
            if lines[0] == None:
                return None
            if lines[0].find("`") == -1 and lines[0].find("Log template:") == -1:
                return lines[0]
        
        return None
    
    def requery_template_from_llm(self, 
                                  log, 
                                  matcher: LogMatcher, 
                                  examples: list, 
                                  model: str, 
                                  filter_ids: list,
                                  knowledgebase: bool,
                                  strategy: int) -> Tuple[str, int, int]:
        """
        Re-query the template from LLM because no template founded.

        returns:
            retry_times: LLM query times
            matcher: the LogMatcher object
        """
        retry_times = 0
        template_id = None
        while retry_times < 3:
            print(f"Querying again, Retrying#{retry_times+1}...")
            query_result = self.query_template_from_llm(log, 
                                                        examples=examples, 
                                                        model=model, 
                                                        knowledgebase=knowledgebase, 
                                                        strategy=strategy)
            print(f"#{retry_times+1} query result is: {query_result}")
            retry_times += 1
            template = self.extract_template_from_llm(query_result)
            if template is not None:
                template = postprocess_template(template)
                if template_id is not None:
                    template_id = matcher.update(log, template, template_id)
                else:
                    template_id = matcher.insert(log, template)

                match_result = matcher.match_template(log, filter_ids)
                if match_result[0]:
                    if template_id == match_result[1]:
                        print(f"Query successfully! The template is: {template}")
                        return template, template_id, retry_times
                    else:
                        filter_ids.append(match_result[1])
        
        print(f"Failed to get correct template from LLM after {retry_times} retries.")
        print("Manual processing...")
        final_template, final_template_id = postprocess_error_template(log, template_id, matcher, filter_ids)
        return final_template, final_template_id, retry_times

    def requery_template_from_llm_with_check(self, 
                                             log, 
                                             template_id, 
                                             matcher: LogMatcher, 
                                             examples: list, 
                                             model: str, 
                                             filter_ids: list,
                                             knowledgebase: bool,
                                             strategy: int) -> Tuple[str, int, int]:
        """
        Re-query the template from LLM, and check for template matching.

        returns:
            template_id: the matching template ID
            retry_times: LLM query times
            matcher: the LogMatcher object
        """
        retry_times = 0
        while retry_times < 3:
            print(f"Querying again, Retrying#{retry_times+1}...")
            query_result = self.query_template_from_llm(log, 
                                                        examples=examples, 
                                                        model=model, 
                                                        knowledgebase=knowledgebase, 
                                                        strategy=strategy)
            print(f"#{retry_times+1} query result is: {query_result}")
            retry_times += 1
            template = self.extract_template_from_llm(query_result)
            if template is not None:
                template = postprocess_template(template)
                template_id = matcher.update(log, template, template_id)
                match_result = matcher.match_template(log, filter_ids)
                if match_result[0]:
                    if template_id == match_result[1]:
                        print(f"Update successfully! The new template is: {template}")
                        return template, template_id, retry_times
                    else:
                        filter_ids.append(match_result[1])
                
        print(f"Failed to get correct template from LLM after {retry_times} retries.")
        print("Manual processing...")
        final_template, final_template_id = postprocess_error_template(log, template_id, matcher, filter_ids)
        return final_template, final_template_id, retry_times

    def requery_template_from_llm_with_erroTemplate(self, 
                                                    log, 
                                                    template_id, 
                                                    matcher: LogMatcher, 
                                                    erroTemplate: str, 
                                                    model: str, 
                                                    filter_ids: list,
                                                    knowledgebase: bool,
                                                    strategy: int) -> Tuple[int, int]:
        """
        Re-query the template from LLM, and check for template matching.

        returns:
            template_id: the matching template ID
            retry_times: LLM query times
            matcher: the LogMatcher object
        """
        retry_times = 0
        while retry_times < 3:
            print(f"Retrying#{retry_times+1}...")
            query_result = self.query_template_from_llm(log, 
                                                        errorMessage=erroTemplate, 
                                                        model=model,
                                                        knowledgebase=knowledgebase,
                                                        strategy=strategy)
            print(f"#{retry_times+1} query result is: {query_result}")
            retry_times += 1
            template = self.extract_template_from_llm(query_result)
            template_extracted = template
            if template is not None:
                template = postprocess_template(template)
                print(f"New template is: {template}")
                template_id = matcher.update(log, template, template_id)
                match_result = matcher.match_template(log, filter_ids)

                if contain_wildcard_only(template):
                    erroTemplate = template_extracted
                    continue

                if match_result[0]:
                    if template_id == match_result[1]:
                        print(f"Update successfully! The new template is: {template}")
                        return template_id, retry_times
                    else:
                        filter_ids.append(match_result[1])
                
        print(f"Failed to get correct template from LLM after {retry_times} retries.")
        print("Manual processing...")
        final_template, final_template_id = postprocess_error_template(log, template_id, matcher, filter_ids)
        return final_template_id, retry_times