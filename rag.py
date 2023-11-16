import os
import pickle
#api key(추가해서 쓰시오)
from workspace.settings import openai_api_key
from workspace.analogicalPrompt import generateAnalogicalPrompt
from workspace.embeddingSetup import EmbeddingLoader

from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import LLMChain, RetrievalQA, HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate


#api key settings
os.environ["OPENAI_API_KEY"] = openai_api_key

#directory settings
directory = os.path.dirname(__file__)
os.chdir(directory)

#llm
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

#embedding config
embedding = EmbeddingLoader().load()

## embedding config - HyDE
hyde_prompt_template = """ 
    Write a passage in Korean to answer the #question in detail.

    #question : {question}
    #passage : ...

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

#get vectorstore *HyDE Embedding
vectorstore = Chroma(collection_name="vector_db", persist_directory="./chroma_storage", embedding_function=hydeembeddings)

#### pickle 받도록 수정해야 함..
#get document from pickle(use as documents in bm25_retriever)
with open('./document.pkl', 'rb') as file :
    documents = pickle.load(file)

#Vector Search Retriever
chroma_retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={"k":5},)

#BM25 Retriever
bm25_retriever = BM25Retriever.from_documents(documents=documents)
bm25_retriever.k = 5

#ensemble
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5],)

#setup chain
chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=ensemble_retriever,
                    chain_type_kwargs={"prompt":generateAnalogicalPrompt()})

##### 예시
response = chain.run("어민 풍수해 피해에 따른 보험금 산정은 어떻게 이루어져?")
print(response)

""" ref
https://python.langchain.com/docs/use_cases/question_answering/vector_db_qa
https://js.langchain.com/docs/modules/chains/popular/vector_db_qa/
https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa
"""