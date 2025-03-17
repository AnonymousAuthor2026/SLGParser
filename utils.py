from LogMatcher import LogMatcher
import pandas as pd
import csv
import os


def cache_to_file(log_tuples, cached_file):
    with open(cached_file, "w") as fw:
        for tuples in log_tuples:
            fw.write(f"{tuples[0]}, {tuples[1]}\n")

def result_to_csv(log_tuples: tuple, template_tuples: tuple, matcher: LogMatcher, file):
    data = []
    
    dir_path = os.path.dirname(file)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for log, template in zip(log_tuples, template_tuples):
        if template[1] != -1:
            data.append({
                "Content": log[1], "EventId": f"E{template[1]}", "EventTemplate": matcher.templates[template[1]]})
        else:
            data.append({"Content": log[1], "EventId": f"E{template[1]}", "EventTemplate": "LLM Parsing Error"})

    df = pd.DataFrame(data)
    df.to_csv(file, index=False, encoding='utf-8')

def evaluate_result_to_csv(dataset, parsing_time, evalute_time, PA, GA, FGA, FTA, candidate,
            similarity_algorithm,
            sample,
            knowledgebase,
            strategy,
            model):
    result_file = f"result_{candidate}_{similarity_algorithm}_{sample}_{knowledgebase}_{strategy}_{model}.csv"
    result = {
        "Dataset": dataset,
        "TotalTime": parsing_time + evalute_time,
        "ParsingTime": parsing_time,
        "EvaluteTime": evalute_time,
        "PA": PA,
        "GA": GA,
        "FGA": FGA,
        "FTA": FTA
    }
    try:
        with open(result_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=result.keys())
            
            if file.tell() == 0:
                writer.writeheader()
            
            writer.writerow(result)
    except Exception as e:
        print(f"[ERROR] Failed to write result to {result_file}: {e}")