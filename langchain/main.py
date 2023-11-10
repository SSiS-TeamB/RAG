""" Dataset generation Example usage """

from document import BaseDBLoader
from qa_generation import generate_qa 
from prompt import da_format_prompt
import json

## corpus 생성까지
loader = BaseDBLoader()

document = loader.load(is_regex=True, is_split=False)
# print(document)
corpus = loader.get_corpus()

with open('result.json', 'w', encoding='utf-8') as file :
    json.dump(corpus, file, indent='\t', ensure_ascii=False)

# dataset = generate_qa(corpus=corpus, prompt_template=da_format_prompt, model="gpt-3.5-turbo-1106", num_questions_per_chunk=3)
# print(dataset)

