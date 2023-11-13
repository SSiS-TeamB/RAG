""" Dataset generation Example usage """

from mdLoaer import BaseDBLoader
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

from qa_generation import generate_qa 
from prompt import prompt_template


## corpus 생성까지
loader = BaseDBLoader()

document = loader.load(is_regex=True, is_split=True)
# print(document)
corpus = loader.get_corpus()
print(corpus)

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