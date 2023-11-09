""" LangChain version QA generation (currently use OpenAI) """

import os
import re
import uuid

from settings import openai_api_key
from prompt import prompt_template

from tqdm.notebook import tqdm
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = openai_api_key

def _write_dataset(node_id:str, questions:list, queries:dict, relevant_docs:dict)->None:
    for question in questions :
        question_id = str(uuid.uuid4())

        queries[question_id] = question
        relevant_docs[question_id] = [node_id]
    
def generate_qa(corpus:dict, prompt_template:str=prompt_template, model:str="gpt-3.5-turbo-1106", num_questions_per_chunk:int=2)->dict:
    dataset = {}
    queries = {}
    relevant_docs = {}
    llm = OpenAI().chat.completions

    for node_id, text in tqdm(corpus.items()) :
        query = prompt_template.format(context_str=text, num_questions_per_chunk=num_questions_per_chunk)

        #llm config(OpenAI API function)
        response = llm.create(
            model = model,
            messages=[{"role":"system","content":query}],
            temperature=1,)

        parsed_response = str(response.choices[0].message.content).strip().split("\n")
        questions=[re.sub(r"^\d+[\).\s]", "", question).strip() for question in parsed_response]
        questions=[question for question in questions if len(question) > 0]

        _write_dataset(node_id, questions, queries, relevant_docs)

    dataset = {'queries':queries, 'corpus':corpus, 'relevant_docs':relevant_docs}
    return dataset

# print(generate_qa(corpus={"1234":"겨울 LPG 난방 지원 사업","11345-123-ab":"참전용사 보훈 지원"}, num_questions_per_chunk=2))