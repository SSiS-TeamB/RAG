""" Dataset generation Example usage """

from document import BaseDBLoader
from qa_generation import generate_qa 
from prompt import prompt_template


## corpus 생성까지
loader = BaseDBLoader()

document = loader.load(is_regex=True, is_split=True)
# print(document)
corpus = loader.get_corpus()
print(corpus)

# dataset = generate_qa(corpus=corpus, prompt_template=prompt_template, model="gpt-3.5-turbo-1106", num_questions_per_chunk=3)
# print(dataset)