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
    'covid': 'beir-v1.0.0-trec-covid.contriever',
    'touche': 'beir-v1.0.0-webis-touche2020.contriever',
    'news': 'beir-v1.0.0-trec-news.contriever',
    'nfc': 'beir-v1.0.0-nfcorpus.contriever',
    'signal': 'beir-v1.0.0-signal1m.contriever',
}

flat_indexes = {
    'dl19': 'msmarco-v1-passage',
    'dl20': 'msmarco-v1-passage',
    'covid': 'beir-v1.0.0-trec-covid.flat',
    'touche': 'beir-v1.0.0-webis-touche2020.flat',
    'news': 'beir-v1.0.0-trec-news.flat',
    'nfc': 'beir-v1.0.0-nfcorpus.flat',
    'signal': 'beir-v1.0.0-signal1m.flat',
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



class Validator:
    def __init__(self, llm_model, tokenizer):
        self.llm_model = llm_model
        self.tokenizer = tokenizer

    def enquire_valid_texts(self, texts, query):
        template_texts =""
        for i, text in enumerate(texts):
            template_texts += f'{i+1}. {text} \n'
        # truncation for large texts
        template_texts = " ".join(template_texts.split(" ")[:2040])
        conversation = [ {'role': 'user', 'content': f'For the following query and document, judge whether they are relevant. Output “Yes” or “No”.\nQuery: {query}\nDocument: {template_texts}'} ]

        prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm_model.device) 
        outputs = self.llm_model.generate(**inputs, use_cache=True, max_length=2048,do_sample=True,temperature=0.7,top_p=0.95,top_k=10,repetition_penalty=1.1)
        output_text = self.tokenizer.decode(outputs[0]) 
        assistant_respond = output_text.split("Assistant:")[1]
        if 'Yes' in assistant_respond:
            return True
        else:
            return False
         

class ValidatorMistral:
    def __init__(self, llm_model, tokenizer):
        self.llm_model = llm_model
        self.tokenizer = tokenizer

    def enquire_valid_texts(self, texts, query):
        template_texts =""
        for i, text in enumerate(texts):
            template_texts += f'{i+1}. {text} \n'
        # truncation for large texts
        template_texts = " ".join(template_texts.split(" ")[:2040])
        conversation = [ {'role': 'user', 'content': f'For the following query and document, judge whether they are relevant. Output “Yes” or “No”.\nQuery: {query}\nDocument: {template_texts}'} ]

        prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm_model.device) 
        outputs = self.llm_model.generate(**inputs, use_cache=True, max_length=2048,do_sample=True,temperature=0.7,top_p=0.95,top_k=10,repetition_penalty=1.1,pad_token_id=self.tokenizer.eos_token_id)
        output_text = self.tokenizer.decode(outputs[0]) 
        assistant_respond = output_text.split("[/INST]")[1]
        if 'Yes' in assistant_respond:
            return True
        else:
            return False
         


def run_retriever(dataset, method, topics, searcher, qrels, expanded_queries, query_encoder, k, text_index, blender=None, validator=None):
    results = []
    for qid in tqdm(topics):
        if qid in qrels:
            question = topics[qid]['title']
            contexts = [cc for cc in expanded_queries[qid]['contexts'] if cc]
            #print(contexts)
            if method == 'hyde':
                all_emb_c = []
                for c in contexts:
                    c_emb = query_encoder.encode(c)
                    all_emb_c.append(np.array(c_emb))
                all_emb_c = np.array(all_emb_c)
                avg_emb_c = np.mean(all_emb_c, axis=0)
                avg_emb_c = avg_emb_c.reshape((1, len(avg_emb_c)))
                hits = searcher.search(avg_emb_c, k)
                rank = 0
                for hit in hits:
                    rank += 1
                    results.append({'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
            elif method == 'genra':
                all_emb_c = []
                for c in contexts:
                    c_emb = query_encoder.encode(c)
                    all_emb_c.append(np.array(c_emb))
                all_emb_c = np.array(all_emb_c)
                avg_emb_c = np.mean(all_emb_c, axis=0)
                avg_emb_c = avg_emb_c.reshape((1, len(avg_emb_c)))
                hits = searcher.search(avg_emb_c, 500)
                hyde_texts = []
                candidate_texts = []
                hit_count = 0
                while len(candidate_texts) < 5:
                    if hit_count < len(hits):
                        json_doc = json.loads(text_index.doc(hits[hit_count].docid).raw())
                        if dataset in ['dl19', 'dl20']:
                            doc_text = json_doc['contents']
                        else:
                            doc_text = json_doc['text']
                        if validator.enquire_valid_texts([doc_text], question):
                            candidate_texts.append(doc_text)
                        hit_count += 1
                    else:
                        break
                # in case no verified doc in topK then return initial q or contexts
                if len(candidate_texts)<1:
                    candidate_texts = contexts #.append(question)
                all_emb_c = []
                all_hits = []
                for i, c in enumerate(candidate_texts):
                    c_emb = query_encoder.encode(c)
                    c_emb = c_emb.reshape((1, len(c_emb)))
                    c_hits = searcher.search(c_emb, k) #
                    all_hits.append(c_hits)
                    rank=0
                    for hit in c_hits:
                        rank += 1
                        results.append({'qid': qid, 'voter':i, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
            elif method == 'genraBM25':
                hits = text_index.search(question, 500)
                hyde_texts = []
                candidate_texts = []
                hit_count = 0
                #print(type(hits))
                while len(candidate_texts) < 5:
                    if hit_count < len(hits):
                        json_doc = json.loads(text_index.doc(hits[hit_count].docid).raw())
                        if dataset in ['dl19', 'dl20']:
                            doc_text = json_doc['contents']
                        else:
                            doc_text = json_doc['text']
                        if validator.enquire_valid_texts([doc_text], question):
                            candidate_texts.append(doc_text)
                        hit_count += 1
                    else:
                        break
                # in case no verified doc in topK then return initial q or contexts
                if len(candidate_texts)<1:
                    candidate_texts = contexts #.append(question)

                for i, c in enumerate(candidate_texts):
                    # truncate long queries for pyserini limit
                    ctrun = " ".join(c.split(" ")[:1000])
                    c_hits = text_index.search(ctrun, k) #
                    rank=0
                    for hit in c_hits: # get top1000 anyway
                        rank += 1
                        results.append({'qid': qid, 'voter':i, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
            elif method == 'baseline_contriever':
                query = question 
                query_emb = query_encoder.encode(query)
                query_emb = query_emb.reshape((1, len(query_emb)))
                hits = searcher.search(query_emb, k)
                rank = 0
                for hit in hits:
                    rank += 1
                    results.append({'qid': qid, 'docid': hit.docid, 'rank': rank, 'score': hit.score})
                #f.write(f'{qid} Q0 {hit.docid} {rank} {hit.score} rank\n')
    return results


def run():
    k = 100
    load_from = 'pickle'
    num_answers = '10'
    runs = ['0'] 
    
    query_encoder = AutoQueryEncoder(encoder_dir='facebook/contriever', pooling='mean')
    validator = None
    llm = 'solar' # mistral or solar
    methods = ['genra','genraBM25'] #['baseline_contriever', 'hyde', 'genra', 'genraBM25']
    gen_data_path = f'data/generated/{llm}_trec_beir/'
    results_path = 'results_trec_beir/'
    for method in methods:

        if method == 'llm_blender':
            llmblender = llm_blender.Blender()
            llmblender.loadranker("llm-blender/PairRM")
        elif method in ['genra', 'genraBM25']:
            if llm == 'solar':
                tokenizer = AutoTokenizer.from_pretrained("Upstage/SOLAR-10.7B-Instruct-v1.0", use_fast=True)
                llm_model = AutoModelForCausalLM.from_pretrained(
                "Upstage/SOLAR-10.7B-Instruct-v1.0",
                device_map="auto",
                torch_dtype=torch.float16,
                )
                validator = Validator(llm_model, tokenizer)
            elif llm == 'mistral':
                tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", use_fast=True)
                llm_model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                device_map="auto",
                torch_dtype=torch.float16,
                )
                validator = ValidatorMistral(llm_model, tokenizer)


        for dataset in ['dl19']: #['dl19', 'dl20', 'covid', 'nfc', 'touche', 'signal', 'news']:
            print('#' * 20)
            print(f'Evaluation on {dataset}')
            print('#' * 20)

            if dataset in ['dl19', 'dl20']:
                searcher = FaissSearcher('data/contriever_msmarco_index/', query_encoder)
                text_index = LuceneSearcher.from_prebuilt_index(flat_indexes[dataset])
            else:
                searcher = FaissSearcher.from_prebuilt_index(dataset_indexes[dataset], query_encoder)
                text_index = LuceneSearcher.from_prebuilt_index(flat_indexes[dataset])
            
            if os.path.exists('datasets_topics_and_qrels.pkl'):
                print("[info:] Topics and qrels already exist! Loading...")
                d_topics_and_qrels = pickle.load(open('datasets_topics_and_qrels.pkl', 'rb'))
                topics = d_topics_and_qrels[dataset]['topics']
                qrels = d_topics_and_qrels[dataset]['qrels']
            else:
                print("[info:] Retrieving Topics and qrels for dataset...")
                topics = get_topics(dataset_topics[dataset] if dataset != 'dl20' else 'dl20')
                qrels = get_qrels(dataset_topics[dataset])

            for run in runs:
                if load_from == 'pickle':
                    expanded_queries = pickle.load(open(gen_data_path+f'{llm}-{dataset}-{num_answers}passages-gen-run{run}.pkl', 'rb'))
                    results, results_h = run_retriever(dataset, method, topics, searcher, qrels, expanded_queries, query_encoder, k, text_index, blender=llmblender, validator=validator)
                elif load_from == 'json':
                    expanded_queries = {}
                    with open(gen_data_path+f'{llm}-{dataset}-{num_answers}passages-gen-run{run}.jsonl') as jf:
                        for row in jf.readlines():
                            rowj = json.loads(row)
                            expanded_queries[rowj['query_id']] = {'query_id': rowj['query_id'], 'query': rowj['query'], 'contexts': rowj['contexts']}

                    results = run_retriever(dataset, method, topics, searcher, qrels, expanded_queries, query_encoder, k, text_index, blender=llmblender, validator=validator)

                results_filename = f'{method}-{dataset}-{llm}-{num_answers}passages-run{run}-results-k{k}'

                write_eval_file(method, dataset, results, results_filename)



def write_eval_file(method, dataset, rank_results, file):
    results_path = 'results_trec_beir/'
    if method in ['genra', 'genraBM25']:
        # write for pyflagr aggregation format
        with open(results_path+file, 'w') as f:
            for res in rank_results:
                f.write(f"{res['qid']},V{res['voter']},{res['docid']},{res['score']},{dataset}\n")
    else:
        with open(results_path+file, 'w') as f:
            for res in rank_results:
                f.write(f"{res['qid']} Q0 {res['docid']} {res['rank']} {res['score']} rank\n")

if __name__ == "__main__":
    run()
