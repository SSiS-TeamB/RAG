import os
import re
import uuid

from settings import openai_api_key
from prompt import prompt_template

from tqdm.notebook import tqdm
from openai import OpenAI

"""
참고 설명: https://wikidocs.net/204333
https://platform.openai.com/docs/guides/text-generation

"""


os.environ["OPENAI_API_KEY"] = openai_api_key


def _write_dataset(node_id: str, questions: list, queries: dict, relevant_docs: dict) -> None:
    for question in questions:
        question_id = str(uuid.uuid4())

        queries[question_id] = question
        relevant_docs[question_id] = [node_id]
    return


def generate_qa(corpus: dict, p_template: str = prompt_template,
                model="gpt-3.5-turbo-1106", questions_per_chunk: int = 2):

    # queries = {}
    relevant_docs = {}
    # llm config(OpenAI API function)
    llm = OpenAI().chat.completions

    for node_id, text in tqdm(corpus.items()):
        # GPT에 던질 프롬프트: 질문 만들어달라는 내용
        prompt = p_template.format(context_str=text, questions_per_chunk=questions_per_chunk)

        # response: GPT로 생성한 질문
        msg = {"role": "system", "content": prompt}
        response = llm.create(model=model, messages=[msg], temperature=1)

        parsed_response = str(response.choices[0].message.content).strip().split("\n")
        questions = [re.sub(r"^\d+[\).\s]", "", question).strip() for question in parsed_response]
        questions = [question for question in questions if len(question) > 0]

        _write_dataset(node_id, questions, queries, relevant_docs)

    dataset = {'queries': queries, 'corpus': corpus, 'relevant_docs': relevant_docs}
    return dataset