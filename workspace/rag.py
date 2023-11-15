import os

from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

#api key(추가해서 쓰시오)
import settings

from langchain.chat_models import ChatOpenAI
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import RetrievalQA, HypotheticalDocumentEmbedder

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from workspace.device_check import device_check

import pickle

os.environ["OPENAI_API_KEY"] = settings.openai_api_key

directory = os.path.dirname(__file__)
os.chdir(directory)

#llm
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=1)

#embedding config
embedding = SentenceTransformerEmbeddings(
    model_name="da_finetune_epoch_2", 
    model_kwargs={'device':device_check()}, 
    encode_kwargs={'normalize_embeddings':True},
    )

## embedding config - HyDE
hyde_prompt_template = """ 
    Write a passage in Korean to answer the #question in detail.

    #question : {question}
    #passage : ...

"""

hyde_prompt_complete = PromptTemplate(input_variables=["question"], template= hyde_prompt_template)
llm_chain = LLMChain(
    llm=llm, 
    prompt=hyde_prompt_complete,
)

HyDEembeddings = HypotheticalDocumentEmbedder(
    llm_chain=llm_chain,
    base_embeddings=embedding,
)

#get vectorstore *HyDE Embedding
vectorstore = Chroma(collection_name="vector_db", persist_directory="./chroma_storage", embedding_function=HyDEembeddings)

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
                    retriever=ensemble_retriever,)

#### 실행단 부분 수정 필요.
result = chain.run("당신은 한국의 복지 전문가입니다. 주어진 정보만을 가지고, 다음의 질문에 대답하면 됩니다. 질문 : 농촌 풍수해 피해의 경우 보상받을 수 있는 방법은? 답변 : ... ")
print(result)


""" ref
https://python.langchain.com/docs/use_cases/question_answering/vector_db_qa
https://js.langchain.com/docs/modules/chains/popular/vector_db_qa/
https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa
"""