""" LangChain version QA generation (currently use OpenAI) """

import os
import re
import uuid

from settings import openai_api_key
from prompt import prompt_template

from tqdm.notebook import tqdm
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = openai_api_key


def _write_dataset(node_id: str, questions: list[str], queries: dict, relevant_docs: dict) -> None:
    for question in questions:
        question_id = str(uuid.uuid4())

        queries[question_id] = question
        relevant_docs[question_id] = [node_id]
    return

def generate_qa(corpus: dict, p_template: str = prompt_template,
                model: str = "gpt-3.5-turbo-1106", questions_per_chunk: int = 2) -> dict:

    llm = OpenAI().chat.completions

    for node_id, text in tqdm(corpus.items()):
        # 프롬프트 $$$
        # https://platform.openai.com/docs/guides/text-generation/chat-completions-api
        prompt = p_template.format(context_str=text, num_questions_per_chunk=questions_per_chunk)
        msg = {"role": "system", "content": prompt}

        # GPT에 프롬프트 넣었을 때 답변 $$$
        response = llm.create(model = model, messages=[msg], temperature=1)
        parsed_response_str = str(response.choices[0].message.content).strip().split("\n")
        questions=[re.sub(r"^\d+[\).\s]", "", question).strip() for question in parsed_response_str]
        questions=[question for question in questions if len(question) > 0]

        # 방법 바꾸고 싶다 $$$
        queries = {}
        relevant_docs = {}
        _write_dataset(node_id, questions, queries, relevant_docs)

    # $$$ queries 원소 채우는 방식 다시 봐야할 듯
    dataset = {'queries': queries, 'corpus': corpus, 'relevant_docs': relevant_docs}
    return dataset

# print(generate_qa(corpus={"1234":"겨울 LPG 난방 지원 사업","11345-123-ab":"참전용사 보훈 지원"}, num_questions_per_chunk=2))
