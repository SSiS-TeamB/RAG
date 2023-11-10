""" LangChain version QA generation (currently use OpenAI) """

import os
import re
import uuid

from settings import openai_api_key

from tqdm.notebook import tqdm
from openai import OpenAI

from prompt import da_format_prompt

from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field

os.environ["OPENAI_API_KEY"] = openai_api_key

def _write_dataset(node_id:str, questions:list, queries:dict, relevant_docs:dict)->None:
    for question in questions :
        question_id = str(uuid.uuid4())

        queries[question_id] = question
        relevant_docs[question_id] = [node_id]

##### 1109 기준 OpenAI 최신버전(1.2.0) 있어야 가동 가능.
def generate_qa(corpus:dict, model:str="gpt-3.5-turbo-1106", num_questions_per_chunk:int=2)->dict:
    dataset = {}
    queries = {}
    relevant_docs = {}
    llm = OpenAI().chat.completions
    
    #outputparser config
    class analogical_thinking(BaseModel):
        reasoning: list[str] = Field(description="예시로 생각한 논리적 구조")
        answer: list[str] = Field(descripton="예제에 따라 생성한 질문")

    pydanticoutput = PydanticOutputParser(pydantic_object=analogical_thinking)

    #main
    for node_id, text in tqdm(corpus.items()) :
        query = da_format_prompt(context=text, num_questions_per_chunk=num_questions_per_chunk)
        # llm config(OpenAI API function)

        response = llm.create(
            model = model,
            temperature=0.2,
            messages=[{"role":"system","content":query}],
            )
        
        output = str(response.choices[0].message.content)
        parsed_output = pydanticoutput.parse(output)

        print(f""" 논리에 따라 생성한 결과는 다음과 같습니다. 
              {parsed_output.reasoning}\n\n{parsed_output.answer}""")

        questions = parsed_output.answer
        # questions=[re.sub(r"^\d+[\).\s]", "", question).strip() for question in parsed_response] #원래 text로 나올 경우 정규식 이용해서 자른 것. (legacy)
        questions=[question for question in questions if len(question) > 0]
        _write_dataset(node_id, questions, queries, relevant_docs)

    dataset = {'queries':queries, 'corpus':corpus, 'relevant_docs':relevant_docs}
    return dataset

##test
# print(generate_qa(corpus={"1234":"겨울 LPG 난방 지원 사업","11345-123-ab":"참전용사 보훈 지원"}, num_questions_per_chunk=3))