import os
import pandas as pd
import time
import re
import calendar
import heapq
import random
from collections import defaultdict, Counter
from sklearn.utils import shuffle
import numpy as np
import json
from tqdm import tqdm
from settings import datasets
from collections import OrderedDict
import Levenshtein
import itertools
import textdistance

class Vocab:
    def __init__(self, stopwords=["<*>"]):
        stopwords = [
            "a",
            "an",
            "and",
            "i",
            "ie",
            "so",
            "to",
            "the",

        ] + list(calendar.day_name) + list(calendar.day_abbr) \
          + list(calendar.month_name) + list(calendar.month_abbr)
        self.token_counter = defaultdict(Counter)
        self.stopwords = frozenset(set(stopwords))

    def build(self, sequences):
        print("\nBuild vocab with examples: ", len(sequences))
        for sequence in sequences:
            seq_length = len(sequence)
            sequence = self.__filter_stopwords(sequence)
            self.update(sequence,seq_length)

    def update(self, sequence,seq_length):
        self.token_counter[seq_length].update(sequence)

    def topk_tokens(self, sequence, topk=3):
        seq_length = len(sequence)
        sequence = self.__filter_stopwords(sequence)
        sequence = list(OrderedDict.fromkeys(sequence))
        word_positions = {token: sequence.index(token) for token in sequence}
        token_count = [(token, self.token_counter[seq_length][token]) for token in sequence]
        topk_tuples = heapq.nlargest(topk, token_count, key=lambda x: (x[1], -word_positions[x[0]]))
        topk_keys = tuple([t[0] for t in topk_tuples])
        return topk_keys

    def __len__(self):
        return len(self.token_counter)

    def __filter_stopwords(self, sequence):
        return [
            token
            for token in sequence
            if (len(token) > 2) and (token not in self.stopwords)
        ]

class LogSampler:
    def __init__(self, dataset):
        self.dataset = dataset

    
    def post_process(self, message):

        # ipv4_pattern = r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(:[0-9]{1,5})?"
        ipv4_pattern = r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
        new_ip = "192.168.1.1"
        new_message = re.sub(ipv4_pattern, new_ip, message)

        hex_pattern = r'0x[0-9A-Fa-f]{8}'
        new_hex = "0xabcdef01"
        new_message = re.sub(hex_pattern, new_hex, new_message)

        longNum_pattern = r'\d{5,}'
        new_longNum = "1234"
        new_message = re.sub(longNum_pattern, new_longNum, new_message)

        domain_pattern = r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}'
        new_domain = ''
        new_message = re.sub(domain_pattern, new_domain, new_message)

        lifetime_pattern = r'(([01]?\d|2[0-5]):[0-5]\d)|(<\d\ssec)'
        new_lifetime = "00:00"
        new_message = re.sub(lifetime_pattern, new_lifetime, new_message)

        return new_message


    def jaccard_similarity(self, log):
        """
        Calculate the jaccard similarity of logs.
        """
        candidates = open(f"sampled_examples_pure/{self.dataset}/samples_v2.txt", "r").readlines()
        similarity_list = []
        new_log = self.post_process(log)
        for candidate in candidates:
            new_candidate = self.post_process(candidate)
            similarity = textdistance.jaccard.normalized_similarity(new_log, new_candidate)
            
            similarity_list.append((similarity, candidate.strip()))
        similarity_list = sorted(similarity_list, key=lambda x: x[0], reverse=True)
        return similarity_list


    def select_examples(self, log):
        examples = self.jaccard_similarity(log)
        examples = [(example[1],example[0]) for example in examples[:5]]

        examples = [example[0] for example in examples if float(example[1]) > 0.85]
        return examples


def preprocess(log):
    log_format = re.sub(r'[0-9A-Za-z,\s]+', '', log)
    unique_chars = list(set(log_format))
    sorted_characters = ''.join(sorted(unique_chars))
    log = re.sub(r'[:()=,"{}@$[\]|;.!?]', ' ', log)
    log = re.sub(r'(/[^\s]+)|([A-Za-z]:\\[^\s]+)',' ',log)
    content = " ".join([word for word in log.strip().split() if not bool(re.search(r'\d+', word))])
    return content, sorted_characters

def hierichical_clustering(contents):
    # build vocab table
    vocab = Vocab()
    vocab.build([v[0].split() for v in contents.values()])
    # hierichical clustering
    hierichical_clusters = {}
    for k, v in tqdm(contents.items()):
        token_length = len(v[0].split())
        # frequent_token = tuple(sorted(vocab.topk_tokens(v[0].split(), 3)))
        frequent_token = vocab.topk_tokens(v[0].split(),3)
        log_format = v[1]
        if token_length not in hierichical_clusters:
            hierichical_clusters[token_length] = {"size": 1, frequent_token : {"size": 1 , log_format:[k]}}
        else:
            hierichical_clusters[token_length]["size"] = hierichical_clusters[token_length]["size"] + 1
            if frequent_token not in hierichical_clusters[token_length]:
                hierichical_clusters[token_length][frequent_token] = {"size": 1, log_format:[k]}
            else:
                hierichical_clusters[token_length][frequent_token]["size"] = hierichical_clusters[token_length][frequent_token]["size"] + 1
                if log_format not in hierichical_clusters[token_length][frequent_token]:
                    hierichical_clusters[token_length][frequent_token][log_format] = [k]
                else:
                    hierichical_clusters[token_length][frequent_token][log_format].append(k)
    
    return hierichical_clusters


def select_most_different(lst, n):

    pairs = itertools.combinations(lst, 2)
    diff_pairs = [(Levenshtein.distance(a[1], b[1]), a[0], b[0], a[1], b[1]) for a, b in pairs]

    # sort by distance
    diff_pairs.sort(key=lambda x: x[0], reverse=True)

    result = set()
    result_idx = []
    for pair in diff_pairs:

        if pair[3] not in result:
            result_idx.append(pair[1])
            result.add(pair[3])

        if pair[4] not in result:
            result_idx.append(pair[2])
            result.add(pair[4])
        
        if len(result_idx) >= n:
            break
    
    if len(result_idx) < n:
        for pair in diff_pairs:

            if pair[1] not in result_idx:
                result_idx.append(pair[1])
            
            if pair[2] not in result_idx:
                result_idx.append(pair[2])
            
            if len(result_idx) >= n:
                break
    
    return result_idx[:n]

def hierichical_distribute(hierichical_clusters, labeled_logs):
    # hierichical distribution
    candidate_samples = []
    first_keys = hierichical_clusters.keys()
    first_keys = shuffle(list(first_keys))
    for _, length_key in enumerate(first_keys):
        
        second_keys = [key for key in hierichical_clusters[length_key].keys() if key != 'size']

        for _, token_key in enumerate(second_keys):

            third_keys = [key for key in hierichical_clusters[length_key][token_key].keys() if key != 'size']

            for _, sign_key in enumerate(third_keys):

                key_lst_len = len(hierichical_clusters[length_key][token_key][sign_key])
                if key_lst_len < 6:
                    samples = hierichical_clusters[length_key][token_key][sign_key]
                else:

                    unselected_list = hierichical_clusters[length_key][token_key][sign_key]

                    unselected_list = random.sample(unselected_list, 50) if len(unselected_list) > 50 else unselected_list
                    
                    unselected_list = [(idx, labeled_logs.loc[idx, 'Content']) for idx in unselected_list]

                    samples = select_most_different(unselected_list, 5)

                candidate_samples.extend(samples)
    return candidate_samples

if __name__ == '__main__':

    for dataset in datasets:
        # hierarchical sampling
        print(f"Performing hierarchical sampling {dataset} ...")
        logs = pd.read_csv(f'full_dataset/{dataset}/{dataset}_full.log_structured.csv')
        labelled_logs = logs.sample(frac=0.3, axis=0).sort_index()
        if not os.path.exists(f"sampled_examples_pure/{dataset}"):
            os.makedirs(f"sampled_examples_pure/{dataset}")
        start_time = time.time()
        contents = {}
        # process log content
        for index, row in labelled_logs.iterrows():
            content = row['Content']
            content, characters = preprocess(content)
            contents[index] = (content, characters)

        hierichical_clusters = hierichical_clustering(contents)

        # with open(f"sampled_examples_pure/{dataset}/hierarchical_samples.json", "w") as f:
        #     for k,v in hierichical_clusters.items():
        #         f.write(f"{k}: {v}\n")

        sampled_ids = hierichical_distribute(hierichical_clusters, labelled_logs)

        candidate_samples = [labelled_logs.loc[id, 'Content'] for id in sampled_ids]
        end_time = time.time()
        print(f"datase[{dataset}] clustering spent: {end_time-start_time} seconds")
        with open(f"sampled_examples_pure/{dataset}/hierarchical_samples.txt", "w") as f:
            for sample in candidate_samples:
                f.write(sample + "\n")

        # random sampling
        print(f"Performing random sampling {dataset} ...")
        random_sample = logs.sample(n=200)
        content_list = random_sample['Content'].tolist()
        output_file = f"sampled_examples_pure/{dataset}/samples_200"
        with open(output_file, 'w', encoding='utf-8') as f:
            for content in content_list:
                f.write(content + '\n')