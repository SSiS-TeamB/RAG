""" Dataset generation Example usage """

from document import RegExDBLodaer
from qa_generation import generate_qa 
from prompt import prompt_template

corpus = RegExDBLodaer().load()

print(corpus)

# dataset = generate_qa(corpus=corpus, prompt_template=prompt_template, model="gpt-3.5-turbo-1106", num_questions_per_chunk=3)

# print(dataset)