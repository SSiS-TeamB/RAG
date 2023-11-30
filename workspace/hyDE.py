###### 시간복잡도 Issue로 HyDE 일단 보류했음. 정확도 측면 제대로 평가하면 쓸 생각.
from langchain.embeddings import HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI


def hyde_embedding_generate(question:str, embedding):
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106")
    
    # embedding config - HyDE
    hyde_prompt_template = """ 
        You will be given a sentence.
        If the sentence is a question, convert it to a plausible answer. 
        If the sentence does not contain a question, 
        just repeat the sentence as is without adding anything to it.

        Examples:
        - what furniture there is in my room? --> In my room there is a bed, 
        a wardrobe and a desk with my computer
        - where did you go today --> today I was at school
        - I like ice cream --> I like ice cream
        - how old is Jack --> Jack is 20 years old

        Answer Only in Korean.
        sentence : {question}
    """

    hyde_prompt = PromptTemplate(input_variables=["question"], template=hyde_prompt_template)

    hyde_generation_chain = LLMChain(
        llm=llm, 
        prompt=hyde_prompt,
    )

    hydeembeddings = HypotheticalDocumentEmbedder(
        llm_chain=hyde_generation_chain,
        base_embeddings=embedding,
    )

# self.rag_chain = (
#             {"context": self.ensemble_retriever | self.format_docs, "question": RunnablePassthrough()}
#             | get_normal_prompt()
#             | self.llm
#             | StrOutputParser()
#         )