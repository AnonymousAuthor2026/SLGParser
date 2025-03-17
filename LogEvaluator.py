import pandas as pd
from tqdm import tqdm
from settings import datasets
import time

class LogEvaluator:
    def __init__(self, dataset):
        self.dataset = dataset

    def calculate_parsing_accuracy(self, groundtruth_df, parsedresult_df):

        correctly_parsed_messages = parsedresult_df[['EventTemplate']].eq(groundtruth_df[['EventTemplate']]).values.sum()
        total_messages = len(parsedresult_df[['Content']])

        return float(correctly_parsed_messages) / total_messages
    
    def calculate_grouping_accuracy(self, groundtruth_df, parsedresult_df):
        groundtruth_series = groundtruth_df['EventTemplate']
        parsedresult_series = parsedresult_df['EventTemplate']

        series_groundtruth_valuecounts = groundtruth_series.value_counts()
        series_parsedlog_valuecounts = parsedresult_series.value_counts()

        df_combined = pd.concat([groundtruth_series, parsedresult_series], axis=1, keys=['groundtruth', 'parsedlog'])
        grouped_df = df_combined.groupby('groundtruth')
        accurate_events = 0
        accurate_templates = 0
        for ground_truthId, group in tqdm(grouped_df):

            series_parsedlog_logId_valuecounts = group['parsedlog'].value_counts()
            
            if series_parsedlog_logId_valuecounts.size == 1:
                parsed_eventId = series_parsedlog_logId_valuecounts.index[0]
                if len(group) == parsedresult_series[parsedresult_series == parsed_eventId].size:
                    accurate_events += len(group)
                    accurate_templates += 1
        GA = float(accurate_events) / len(groundtruth_series)
        PGA = float(accurate_templates) / len(series_parsedlog_valuecounts)
        RGA = float(accurate_templates) / len(series_groundtruth_valuecounts)
        FGA = 0.0
        if PGA != 0 or RGA != 0:
            FGA = 2 * (PGA * RGA) / (PGA + RGA)
        return GA, FGA

    def evaluate_template_level_accuracy(self, df_groundtruth, df_parsedresult):
        correct_parsing_templates = 0
        
        null_logids = df_groundtruth[~df_groundtruth['EventTemplate'].isnull()].index
        df_groundtruth = df_groundtruth.loc[null_logids]
        df_parsedresult = df_parsedresult.loc[null_logids]
        series_groundtruth = df_groundtruth['EventTemplate']
        series_parsedlog = df_parsedresult['EventTemplate']
        series_groundtruth_valuecounts = series_groundtruth.value_counts()
        # series_parsedlog_valuecounts = series_parsedlog.value_counts()

        df_combined = pd.concat([series_groundtruth, series_parsedlog], axis=1, keys=['groundtruth', 'parsedlog'])
        grouped_df = df_combined.groupby('parsedlog')
        
        for identified_template, group in tqdm(grouped_df):
            corr_oracle_templates = set(list(group['groundtruth']))

            if corr_oracle_templates == {identified_template}:
                correct_parsing_templates += 1

        PTA = float(correct_parsing_templates) / len(grouped_df)
        RTA = float(correct_parsing_templates) / len(series_groundtruth_valuecounts)
        FTA = 0.0
        if PTA != 0 or RTA != 0:
            FTA = 2 * (PTA * RTA) / (PTA + RTA)
        return PTA, RTA, FTA

    def evaluate(self, groundtruth_df, parsedresult_df):
        start_time = time.time()
        PA = self.calculate_parsing_accuracy(groundtruth_df, parsedresult_df)

        GA, FGA = self.calculate_grouping_accuracy(groundtruth_df, parsedresult_df)

        PTA, RTA, FTA = self.evaluate_template_level_accuracy(groundtruth_df, parsedresult_df)
        evalute_time = time.time() - start_time
        return PA, GA, FGA, FTA, evalute_time

if __name__ == '__main__':
    
    for dataset in datasets:
        groundtruth_path = f"full_dataset/{dataset}/{dataset}_full.log_structured_new.csv"
        parsedresult_path = f"result/{dataset}/log_structured_3_purelog_gpt-4o-2024-08-06.csv"
        groundtruth_df = pd.read_csv(groundtruth_path)
        parsedresult_df = pd.read_csv(parsedresult_path)
        evaluator = LogEvaluator(dataset)
        PA = evaluator.calculate_parsing_accuracy(groundtruth_df, parsedresult_df)
        print(f"\nThe PA of {dataset} is : {PA}")
        
        GA, FGA = evaluator.calculate_grouping_accuracy(groundtruth_df, parsedresult_df)
        print(f"The GA of {dataset} is : {GA}")
        print(f"The FGA of {dataset} is : {FGA}")

        PTA, RTA, FTA = evaluator.evaluate_template_level_accuracy(groundtruth_df, parsedresult_df)
        # print(f"The PTA of {dataset} is : {PTA}")
        # print(f"The RTA of {dataset} is : {RTA}")
        print(f"The FTA of {dataset} is : {FTA}")

