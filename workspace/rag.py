""" 수정 필요한 부분 -> 
pickle 불러오는 것 (document)
db path에 맞도록 수정
"""

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

from device_check import device_check

import pickle

os.environ["OPENAI_API_KEY"] = settings.openai_api_key

#path 설정 임시(231114)
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
prompt_template = """ 
당신은 대한민국의 복지제도 전문가입니다. 복지 제도를 기반으로, 주어진 #질문에 #답변하면 됩니다.

#질문 : {question}
#답변 : ... 
"""

prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
llm_chain = LLMChain(llm=llm, prompt=prompt)

HyDEembeddings = HypotheticalDocumentEmbedder(
    llm_chain=llm_chain,
    base_embeddings=embedding,
)

#get vectorstore *HyDE Embedding
vectorstore = Chroma(collection_name="vector_db", persist_directory="./chroma_storage", embedding_function=HyDEembeddings)

# print(vectorstore.similarity_search("국가장학금", k=3,))

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


#### prompt 수정 필요
result = chain.run("40대 주거 욕구에 대해 말해줘.")
print(result)


""" ref
https://python.langchain.com/docs/use_cases/question_answering/vector_db_qa
https://js.langchain.com/docs/modules/chains/popular/vector_db_qa/
https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa
"""