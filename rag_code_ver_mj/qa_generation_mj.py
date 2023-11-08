""" LangChain version QA generation (currently use OpenAI) """

import os
import re
import uuid

## api key설정 
from settings import openai_api_key
## prompt 틀 가져오기
from prompt import prompt_template

## task progress 확인을 위한 라이브러리
from tqdm.notebook import tqdm
## Gpt 사용을 위한 library
from openai import OpenAI

## openAPI 사용을 위한 파이썬 환경변수 설정
os.environ["OPENAI_API_KEY"] = openai_api_key


## query와 연관된 document에  질문의 uuid -> question_id, corpus 의 uuid -> node_id 매칭 ==> {question_id : node_id}
def _write_dataset(node_id:str, questions:list, queries:dict, relevant_docs:dict)->None:
    for question in questions :
        question_id = str(uuid.uuid4())  #질문마다 uuid부여

        ## {질문 uuid : 질문} 쌍으로 queries 에 update
        queries[question_id] = question  
        ## {질문 uuid : document uuid(corpus)} 쌍으로 relevant_docs 에 update -> 해당 답변들어왔을때 참조할 document 찾기위함
        relevant_docs[question_id] = [node_id]


## 복지서비스 pdf - corpus를 기반으로 prompting을 통한 QA 생성
def generate_qa(corpus:dict, prompt_template:str=prompt_template, model:str="gpt-3.5-turbo-1106", num_questions_per_chunk:int=2):
    ## 저장할 메모리 확보
    dataset = {}  
    queries = {}
    relevant_docs = {}
    llm = OpenAI().chat.completions  #사용할 chat_model

    # corpus 별로 uuid -> node_id, document 내용 -> text 변환후 , text 별로 query 생성
    # 만들 질문 prompting : prompt_template 사용
    # 만들 질문 개수 : num_questions_per_chunk
    for node_id, text in tqdm(corpus.items()) :
        query = prompt_template.format(context_str=text, num_questions_per_chunk=num_questions_per_chunk)

        #llm config(OpenAI API function)
        response = llm.create(
            model = model, #사용할 모델
            messages=[{"role":"system","content":query}],  #prompting 역할
            temperature=1,)  # temperatue : 응답 창의성 parameter  :::  0(창의성 X) ~ 1(창의성 최대)
                                                                 # temperature = 0 : 일관된 답변  , temperature =1 : 창의성 max

        ## response 에서 question 부분만 parsing 후 , 정규표현식으로 정제후 questions list에 저장
        parsed_response = str(response.choices[0].message.content).strip().split("\n")
        questions=[re.sub(r"^\d+[\).\s]", "", question).strip() for question in parsed_response]
        questions=[question for question in questions if len(question) > 0]

        ## {질문 uuid : document uuid(corpus)} 쌍으로 묶기  : 질문 - document 연결
        _write_dataset(node_id, questions, queries, relevant_docs)

    # queries ==> {질문 uuid : 질문}
    # corpus ==> {documnet uuid : document}
    # relevant_docs ==> {질문 uuid : document uuid}
    dataset = {'queries':queries, 'corpus':corpus, 'relevant_docs':relevant_docs}
    return dataset

# print(generate_qa(corpus={"1234":"겨울 LPG 난방 지원 사업","11345-123-ab":"참전용사 보훈 지원"}, num_questions_per_chunk=2))