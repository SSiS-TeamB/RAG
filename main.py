""" Generate Synthetic dataset with LLM(Chat-gpt) 

    Domain Adaption, Fine Tunning을 위한 데이터셋 생성을 llama-index로 구현. """

import json

from corpus import create_corpus
from qa_generation import generate_qa

corpus = create_corpus()
dataset = generate_qa(corpus=corpus, num_questions_per_chunk=3)

dataset_path = "trainset.json"
with open(dataset_path, "w+") as file:
        json.dump(dataset, file)

