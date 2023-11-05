import re
import uuid
from tqdm.notebook import tqdm
from prompt_config import prompt_template
from llama_index.llms import OpenAI

import json

def generate(corpus:dict, dataset_path:str, prompt_template:str=prompt_template, model:str='gpt-3.5-turbo', num_questions_per_chunk:int=2):
    queries={}
    relevant_docs={}
    llm = OpenAI(model)

    ## tqdm 그냥 iteration progress 보려고 넣은거임 별 의미 없음.
    for node_id, text in tqdm(corpus.items()):
        query = prompt_template.format(context_str=text, num_questions_per_chunk=num_questions_per_chunk)

        response = llm.complete(query)
        parsed_response = str(response).strip().split("\n")

        questions=[re.sub(r"^\d+[\).\s]", "", question).strip() for question in parsed_response]

        questions=[question for question in questions if len(question) > 0]

        for question in questions:
            qusetion_id = str(uuid.uuid4())
            queries[qusetion_id] = question
            relevant_docs[qusetion_id] = [node_id]

    #make dataset, dump to json
    dataset={
    'queries': queries,
    'corpus': corpus,
    'relevant_docs': relevant_docs,
    }
    
    with open(dataset_path, "w+") as file:
        json.dump(dataset, file)