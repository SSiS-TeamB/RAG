""" Dataset generation Example usage """

from document import CorpusDBLoader
from qa_generation import generate_qa 
from prompt import prompt_template

corpus = CorpusDBLoader().load()

import json

with open("result.json", "w", encoding="utf-8") as file :
    json.dump(corpus, file, indent="\t", ensure_ascii=False)

# 5258 corpus generated.
# print(len(corpus))

# dataset = generate_qa(corpus=corpus, prompt_template=prompt_template, model="gpt-3.5-turbo-1106", num_questions_per_chunk=1)