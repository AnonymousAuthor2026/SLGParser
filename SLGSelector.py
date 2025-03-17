import pandas as pd
import re
import calendar
import heapq
from collections import defaultdict, Counter
from settings import datasets, filter_word
from collections import OrderedDict
import Levenshtein
import itertools
import textdistance
import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

class Vocab:
    def __init__(self, dataset):
        stopwords = [
            "a", "an", "and", "i",
            "ie", "so", "to", "the",
            "of", "is", "as", "in",
            "at", "on", "or", "has",
            "yes", "no"
        ] + list(calendar.day_name) + list(calendar.day_abbr) \
          + list(calendar.month_name) + list(calendar.month_abbr) \
          + list(filter_word[dataset]["words"])
        self.token_counter = Counter()
        self.stopwords = frozenset(set(stopwords))


    def build(self, sequences):
        print("\nBuild vocab with examples: ", len(sequences))
        for sequence in sequences:
            sequence = self.__filter_stopwords(sequence)
            self.update(sequence)

    def update(self, sequence):
        self.token_counter.update(sequence)

    def topk_tokens(self, sequence, topk=3):
        sequence = self.__filter_stopwords(sequence)
        sequence = list(OrderedDict.fromkeys(sequence))
        word_positions = {token: sequence.index(token) for token in sequence}
        
        token_count = [(token, self.token_counter[token]) for token in sequence]
        topk_tuples = heapq.nlargest(topk, token_count, key=lambda x: (x[1], -word_positions[x[0]]))
        topk_keys = tuple([t[0] for t in topk_tuples])
        return topk_keys

    def __len__(self):
        return len(self.token_counter)

    def __filter_stopwords(self, sequence):
        
        return [
            token
            for token in sequence
            if (token not in self.stopwords) and (token.lower() not in self.stopwords) and (len(token) > 1)
        ]
    

class SLGSelector:
    def __init__(self, dataset, candidate, similarity_algorithm):
        self.dataset = dataset
        self.candidate_method = candidate
        self.similarity_algorithm = similarity_algorithm
        
        if candidate == 'hierarchical':
            self.candidates = open(f'sampled_examples_pure/{self.dataset}/hierarchical_samples.txt','r').readlines()
        else:
            self.candidates = open(f"sampled_examples_pure/{self.dataset}/samples_200", "r").readlines()
        
        if similarity_algorithm == 'enhanced':
            self.groups = self.grouping()
    
    def post_process(self, message):

        ipv4_pattern = r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
        new_message = re.sub(ipv4_pattern, '', message)

        hex_pattern = r'0x[0-9A-Fa-f]{8}'
        new_message = re.sub(hex_pattern, '', new_message)

        longNum_pattern = r'\d{5,}'
        new_message = re.sub(longNum_pattern, '', new_message)

        path_pattern = r'\/([a-zA-Z0-9-_\/]+(\.[a-zA-Z0-9-_]+)?)'
        new_message = re.sub(path_pattern, '', new_message)

        domain_pattern = r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}'
        new_message = re.sub(domain_pattern, '', new_message)

        lifetime_pattern = r'(([01]?\d|2[0-5]):[0-5]\d)|(<\d\ssec)'
        new_message = re.sub(lifetime_pattern, '', new_message)

        date_pattern = r'\w{3} \w{3} \d{1,2} \d{2}:\d{2}:\d{2} \d{4}'
        new_message = re.sub(date_pattern, '', new_message)

        digit_pattern = r'\d+'
        new_message = re.sub(digit_pattern, '', new_message)

        return new_message

    def filter_words(self, text: str):
        # filter IP type
        text = re.sub(r'\/?(\d{1,3}\.){3}\d{1,3}(:\d{1,5})?', '', text) #ipv4
        text = re.sub(r'([0-9a-fA-F]{1,4}:){1,7}([0-9a-fA-F]{1,4})?::?([0-9a-fA-F]{1,4}:){0,6}([0-9a-fA-F]{1,4})', '', text) #ipv6
        text = re.sub(r'\.{2,}', ' ', text)
        for reg in filter_word[self.dataset]["regs"]:
            text = re.sub(reg, '', text)

        # filter path
        text = re.sub(r'\/([a-zA-Z0-9-_\/]+(\.[a-zA-Z0-9-_]+)?)', '', text)
        # filter url
        text = re.sub(r'https?:\/\/[a-zA-Z0-9\-_.]+(?:\.[a-zA-Z0-9\-_.]+)*(:\d+)?(\/[^\s]*)?', '', text)
        # filter domain type
        text = re.sub(r'(\w+\.)+[\w+\.[a-zA-Z]{2,}', '', text)
        # filter signals
        text = re.sub(r'[_/\'"{}|;?*<>]', '', text)
        text = re.sub(r'[.:=()+$@#,[\]]', ' ', text)
        #  filter hex
        text = re.sub(r'0[xX][0-9A-Fa-f]{1,16}', '', text)
        text = re.sub(r'([0-9A-Fa-f]{2}:){11}[0-9A-Fa-f]{2}', '', text)
        new_text = " ".join([word for word in text.strip().split() if not bool(re.match(r'^[\d]+$', word))])
        return new_text

    def group_similarity(self, log, entries):
        new_log = self.post_process(log)
        similarity = 0.0
        for entry in entries:
            similarity += textdistance.jaccard.normalized_similarity(new_log, self.post_process(entry))
        return similarity/len(entries)

    def grouping(self):
        print("Grouping...")
        samples = self.candidates
        sample_set = set()
        for sample in samples:
            sample = self.filter_words(sample)
            sample_set.add(sample)
        vocab = Vocab(self.dataset)
        vocab.build([sample.split() for sample in sample_set])

        group = defaultdict(list)
        for idx, sample in enumerate(samples):
            sample = self.filter_words(sample)
            frequent_token = vocab.topk_tokens(sample.split(), filter_word[self.dataset]["k"])
            group[frequent_token].append(idx)

        with open(f"sampled_examples_pure/{self.dataset}/group_{self.candidate_method}_{self.similarity_algorithm}.txt",'w') as f:
            for key, ids in group.items():
                f.write(f"{key}\n")
                for id in ids:
                    f.write(f"{samples[id]}")
                f.write("\n")
        
        print(len(group.keys()))
        with open(f"sampled_examples_pure/{self.dataset}/group_{self.candidate_method}_{self.similarity_algorithm}.pkl", 'wb') as f:
            pickle.dump(group, f)
        return group


    def enhanced_sample(self, log, top_n=3):
        selected_group = {}
        for key, ids in self.groups.items():
            group_entries = []
            for id in ids:
                group_entries.append(self.candidates[id].strip())
            group_similarity = self.group_similarity(log, group_entries)
            selected_group[(key, group_similarity)] = group_entries.copy()

        max_group_key = max(selected_group.keys(), key=lambda x: x[1])
        selected_entries = selected_group[max_group_key]

        if len(selected_entries) == 1:
            return [] if selected_entries[0] == log else selected_entries[0]
        
        if len(set(selected_entries)) == 1:
            return []
        
        # return random.sample(selected_entries, min(len(selected_entries), 3))
        selected_entries = [entry for entry in selected_entries if len(entry.split()) == len(log.split())]
        result_list = find_most_dissimilar(log, set(selected_entries), min(len(selected_entries), top_n))
        return result_list
    

    def jaccard_sample(self, log, top_n=3):
        
        def calculate_similarity(item):
            similarity = textdistance.jaccard.normalized_similarity(log, item.strip())
            return item, similarity

        with ThreadPoolExecutor() as executor:
            similarity_list = list(executor.map(calculate_similarity, self.candidates))
        
        result_list = [item.strip() for item, _ in sorted(similarity_list, key=lambda x: x[1], reverse=True)[:top_n]]
        
        return result_list

    def cosine_sample(self, log, top_n=3):
        def calculate_similarity(item):
            similarity = textdistance.cosine.normalized_similarity(log, item.strip())
            return item, similarity

        with ThreadPoolExecutor() as executor:
            similarity_list = list(executor.map(calculate_similarity, self.candidates))
        
        result_list = [item.strip() for item, _ in sorted(similarity_list, key=lambda x: x[1], reverse=True)[:top_n]]
        
        return result_list


    def select_examples(self, log, k=3):
        if self.similarity_algorithm == "enhanced":
            return self.enhanced_sample(log, k)
        elif self.similarity_algorithm == "jaccard":
            return self.jaccard_sample(log, k)
        else:
            return self.cosine_sample(log, k)


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def dissimilar_filter(string):
    string = re.sub(r'[A-Z][a-z]{2}\s[A-Z][a-z]{2}\s\d{1,2}\s\d{2}:\d{2}:\d{2}\s\d{4}', '', string)
    return string

def find_most_dissimilar(log, strings, top_n):
    distances1 = {}
    for s1, s2 in itertools.combinations(strings, 2):
        _s1 = dissimilar_filter(s1)
        _s2 = dissimilar_filter(s2)
        # distance = levenshtein_distance(_s1, _s2)
        distances1[(s1, s2)] = textdistance.jaccard.normalized_distance(_s1, _s2)
    sorted_distances1 = sorted(distances1.items(), key=lambda x: x[1], reverse=True)
    most_dissimilar1 = []
    for pair, distance in sorted_distances1:
        if pair[0] not in most_dissimilar1:
            most_dissimilar1.append(pair[0])
        if pair[1] not in most_dissimilar1:
            most_dissimilar1.append(pair[1])
        if len(most_dissimilar1) >= 2*top_n:
            break
    
    distances2 = {}
    for string in most_dissimilar1:
        _log = dissimilar_filter(log)
        _string = dissimilar_filter(string)
        distances2[(log, string)] = levenshtein_distance(_log, _string)
    sorted_distances2 = sorted(distances2.items(), key=lambda x: x[1], reverse=True)
    most_dissimilar2 = []
    for pair, distance in sorted_distances2:
        if (pair[0] not in most_dissimilar2) and (pair[0] != log):
            most_dissimilar2.append(pair[0])
        if pair[1] not in most_dissimilar2 and (pair[1] != log):
            most_dissimilar2.append(pair[1])
        if len(most_dissimilar2) >= top_n:
            break

    return most_dissimilar2[:top_n]



if __name__ == '__main__':

    for dataset in datasets:
        # samples = open(f'sampled_examples_pure/{dataset}/samples_v2.txt','r').readlines()
        candidate = "hierarchical"
        similarity_algorithm = "enhanced"
        sampler = SLGSelector(dataset, candidate, similarity_algorithm)


    # for dataset in datasets:
    #     sampler = SLGSelector(dataset)
    #     logs = open(f"sampled_examples_pure/{dataset}/{dataset}_simulated.txt", 'r').readlines()
    #     with open(f"sampled_examples_pure/{dataset}/{dataset}_samples.txt", 'w') as f:
    #         for log in logs:
    #             exmaples = sampler.select_examples(log.strip(), "purelog", 3)
    #             f.write(f"The Log is `{log.strip()}`\nExamples: {exmaples}\n\n")
                