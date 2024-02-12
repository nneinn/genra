
import torch
import time
import numpy as np
import pandas as pd
import pickle
import json
import gzip
import ir_datasets
import re
import os

from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertForMaskedLM, BertTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, pipeline

from pyserini.search import FaissSearcher, LuceneSearcher
from pyserini.search.faiss import AutoQueryEncoder
from pyserini.search import get_topics, get_qrels


dataset_indexes = {
    'dl19': 'msmarco-v1-passage',
    'dl20': 'msmarco-v1-passage',
    'covid': 'beir-v1.0.0-trec-covid.contriever-msmarco',
    'touche': 'beir-v1.0.0-webis-touche2020.contriever-msmarco',
    'news': 'beir-v1.0.0-trec-news.contriever-msmarco',
    'nfc': 'beir-v1.0.0-nfcorpus.contriever-msmarco',
    'signal': 'beir-v1.0.0-signal1m.contriever-msmarco',
}

dataset_topics  = {
    'dl19': 'dl19-passage',
    'dl20': 'dl20-passage',
    'covid': 'beir-v1.0.0-trec-covid-test',
    'touche': 'beir-v1.0.0-webis-touche2020-test',
    'news': 'beir-v1.0.0-trec-news-test',
    'nfc': 'beir-v1.0.0-nfcorpus-test',
    'signal': 'beir-v1.0.0-signal1m-test',
}

num_map = {
    1:"one",
    2:"two",
    5:"five",
    10:"ten",
    20:"twenty",
}


class QueryExpander:
    def __init__(self, llm_model, tokenizer, num_answers):
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.num_answers = num_map[num_answers]

    def expand_query(self, query):
        conversation = [ {'role': 'user', 'content': 'Write '+self.num_answers+' different passages for the following topic: '+query+'. Provide your answer as a numbered list.'} ] 

        prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm_model.device) 
        outputs = self.llm_model.generate(**inputs, use_cache=True, max_length=512,do_sample=True,temperature=0.7,top_p=0.95,top_k=10,repetition_penalty=1.1)
        output_text = self.tokenizer.decode(outputs[0]) 
        assistant_respond = output_text.split("Assistant:")[1]
        assistant_splitted = self.split_phrases(assistant_respond)
        return assistant_splitted
    
    def split_phrases(self, response):
        response = response.split('GPT4')[0]
        regex_pattern = r"\d(?<=\d)[\.\)]\s{1}" #r"(?<=\b\d\))\s"
        response_text = re.split(regex_pattern, response)
        splitted_sents = [t.strip() for t in response_text]
        return splitted_sents


def expand_queries_json(llm_name, dataset, num_runs, q_expander, num_answers, topics, qrels):
    path = f'data/generated/{llm_name}_trec_beir/'
    for run in range(num_runs):
        srun = str(run)
        with open(path+f'{llm_name}-{dataset}-{num_answers}passages-gen-run{srun}.jsonl', 'w') as fgen:
            for qid in tqdm(topics):
                if qid in qrels:
                    query = topics[qid]['title']
                    expanded_queries = q_expander.expand_query(query)
                    contexts = [c.strip() for c in expanded_queries] + [query]
                    fgen.write(json.dumps({'query_id': qid, 'query': query, 'contexts': contexts})+'\n')

def expand_queries_pickle(llm_name, dataset, num_runs, q_expander, num_answers, topics, qrels):
    path = f'data/generated/{llm_name}_trec_beir/'
    for run in range(num_runs):
        srun = str(run)
        print("[info:] Generating run ", srun)
        filename = f'{llm_name}-{dataset}-{num_answers}passages-gen-run{srun}.pkl'
        qids_expansion = {}
        for qid in tqdm(topics):
            if qid in qrels:
                query = topics[qid]['title']
                expanded_queries = q_expander.expand_query(query)
                contexts = [c.strip() for c in expanded_queries if c] + [query]
                qids_expansion[qid] = {'query_id': qid, 'query': query, 'contexts': contexts}
        pickle.dump(qids_expansion, open(path+filename, 'wb'))

def main():
    num_runs = 3 
    num_answers = 10
    saveto = 'pickle'
    # initialize llm model
    llm_name = 'solar'
    tokenizer = AutoTokenizer.from_pretrained("Upstage/SOLAR-10.7B-Instruct-v1.0", use_fast=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
    "Upstage/SOLAR-10.7B-Instruct-v1.0",
    device_map="auto",
    torch_dtype=torch.float16,
    )
    
    num_answers_str = str(num_answers)
    q_expander = QueryExpander(llm_model, tokenizer, num_answers)
    for dataset in ['dl19', 'dl20', 'covid', 'nfc', 'touche', 'signal', 'news']:
        print('#' * 20)
        print(f'Expanding Queries on {dataset}')
        print('#' * 20)

        # Retrieve passages using pyserini BM25.
        if os.path.exists('datasets_topics_and_qrels.pkl'):
            print("[info:] Topics and qrels already exist! Loading...")
            d_topics_and_qrels = pickle.load(open('datasets_topics_and_qrels.pkl', 'rb'))
            topics = d_topics_and_qrels[dataset]['topics']
            qrels = d_topics_and_qrels[dataset]['qrels']
        else:
            print("[info:] Retrieving Topics and qrels for dataset...")
            topics = get_topics(dataset_topics[dataset] if dataset != 'dl20' else 'dl20')
            qrels = get_qrels(dataset_topics[dataset])

        if saveto == 'json':
            expand_queries_json(llm_name,dataset, num_runs, q_expander, num_answers_str, topics, qrels)
        else:
            expand_queries_pickle(llm_name,dataset, num_runs, q_expander, num_answers_str, topics, qrels)


if __name__ == "__main__":
    # expand queries 
    main()
