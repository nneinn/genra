# GENRA: Enhancing Zero-shot Retrieval with Rank Aggregation
This repository contains the code for the paper: GENRA: Enhancing Zero-shot Retrieval with Rank Aggregation.

## Prerequisites

- Python >= 3.8
- Nvidia GPU
- torch >= 2.1.2
- transformers >= 4.36.2
- pyserini >= 0.24.0
- pyflagr >= 1.0.8

## Run

To run GENRA on TREC or BEIR datasets you have to execute the following steps:

- Configure and run the `expand_queries.py` script to generate passages for each dataset.
- Configure and run the `trec_beir_exp.py` script. This script will produce a ranking, as a txt file, for each method. For GENRA, the produced file is in the PyFlagr format and needs further processing.
- Configure and run the `aggregate.py` script. This will produce the final aggregated ranking for GENRA.

## Evaluate

Once you have produced the files containing the rankings, you can evaluate them using the pyserini trec_eval script as:

```
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 dl19-passage genra-dl19-solar-10passages-run0-results-k100-RA-linear


```
or

```
python -m pyserini.eval.trec_eval -c -l 2 -m recall.100 -m map dl19-passage genra-dl19-solar-10passages-run0-results-k100-RA-linear


```
