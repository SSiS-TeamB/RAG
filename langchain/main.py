""" Dataset generation Example usage """

from document import BaseDBLoader
<<<<<<< HEAD
# from qa_generation import generate_qa
# from prompt import prompt_template

# corpus = RegExDBLodaer().load()
corpus1 = BaseDBLoader().load()
corpus2 = BaseDBLoader().load(is_regex=True)

idx = 0
for e in corpus1:
    print('-'*30)
    print(idx, e)
    print('-'*30)
    if idx > 3:
        break
    idx += 1
=======
from qa_generation import generate_qa 
from prompt import prompt_template
import json

## corpus 생성까지
loader = BaseDBLoader()

document = loader.load(is_regex=True, is_split=False)
# print(document)
corpus = loader.get_corpus()
<<<<<<< HEAD
print(corpus)
>>>>>>> origin/joonho
=======

with open('result.json', 'w', encoding='utf-8') as file :
    json.dump(corpus, file, indent='\t', ensure_ascii=False)
>>>>>>> 60d4caaca32d5be3632e7ca18a91181d02bed710

idx = 0
for e in corpus2:
    print('-'*30)
    print(idx, e)
    print('-'*30)
    if idx > 3:
        break
    idx += 1

print(len(corpus1), len(corpus2))
# dataset = generate_qa(corpus=corpus, prompt_template=prompt_template, model="gpt-3.5-turbo-1106", num_questions_per_chunk=3)
# print(dataset)

