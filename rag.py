import os
import re
import pickle
#api key(추가해서 쓰시오)
from workspace.settings import openai_api_key
from workspace.analogicalPrompt import generateAnalogicalPrompt
from workspace.embeddingSetup import EmbeddingLoader

from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser

###### 시간복잡도 Issue로 HyDE 일단 보류했음. 정확도 측면 제대로 평가하면 쓸 생각.
## embedding config - HyDE
# hyde_prompt_template = """ 
#     Write a passage in Korean to answer the #question in detail.

#     #question : {question}
#     #passage : ...
# """

# hyde_prompt = PromptTemplate(input_variables=["question"], template=hyde_prompt_template)

# hyde_generation_chain = LLMChain(
#     llm=llm, 
#     prompt=hyde_prompt,
# )

# hydeembeddings = HypotheticalDocumentEmbedder(
#     llm_chain=hyde_generation_chain,
#     base_embeddings=embedding,
# )

####################### code 정리하시오

#api key settings
os.environ["OPENAI_API_KEY"] = openai_api_key

########### object화 해서 llm config, chain, 결과 wrap
# def temp_rag_pipeline(query:str) -> str:

#     llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)

#     embedding = EmbeddingLoader().load()
#     vectorstore = Chroma(collection_name="wf_schema", persist_directory="workspace/chroma_storage", embedding_function=embedding)

#     #get document from pickle(use as documents in bm25_retriever)
#     with open('workspace/document.pkl', 'rb') as file :
#         documents = pickle.load(file)

#     #Vector Search Retriever config
#     chroma_retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={"k":5})

#     #BM25 Retriever config
#     bm25_retriever = BM25Retriever.from_documents(documents=documents)
#     bm25_retriever.k = 5

#     #ensemble retrievers
#     ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5],)

#     # LCEL
#     def format_docs(docs):
#         return "\n\n".join(doc.page_content for doc in docs)

#     rag_chain = (
#         {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
#         | generateAnalogicalPrompt()
#         | llm
#         | StrOutputParser()
#     )

#     return rag_chain.invoke(query)

class RAGPipeline:
    def __init__(self, vectorstore, embedding):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)
        self.vectorstore = vectorstore
        self.embedding = embedding
        
        # pickle list 객체 생성 시 로드
        with open('workspace/document.pkl', 'rb') as file:
            self.documents = pickle.load(file)

        # Vector Search Retriever
        self.chroma_retriever = self.vectorstore.as_retriever(search_type='mmr', search_kwargs={"k":5})

        # BM25 Retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents=self.documents)
        self.bm25_retriever.k = 5

        # Ensemble
        self.ensemble_retriever = EnsembleRetriever(retrievers=[self.bm25_retriever, self.chroma_retriever], weights=[0.5, 0.5])

        # RAG 체인 구성
        self.rag_chain = (
            {"context": self.ensemble_retriever | self.format_docs, "question": RunnablePassthrough()}
            | generateAnalogicalPrompt()
            | self.llm
            | StrOutputParser()
        )

    @staticmethod
    def format_docs(docs):
        
        ## 어느 제도 부분에서 가져왔는지 나타내는 출처 : medata 활용해서 같이 출력
        result = []
        for doc in docs:
            title_resource = doc.metadata
            title_resource = str(title_resource).split(":")[1].lstrip()
            title_resource = re.sub(pattern="}|'",repl="",string=title_resource)
            title_resource = title_resource.split("\\")[4]+"_"+title_resource.split("\\")[-1]
            content = doc.page_content 
            # print(f"** 출처 ** : \n {title_resource}", end="\n\n")
            # print(f"*** 제도내용 *** : \n {content}", end="\n\n")
            unit_doc = content + f"\n\n 출처 : {title_resource}"
            result.append(unit_doc)
        
        return "\n\n".join(doc for doc in result)
        


        # # 원래 꺼
        # return "\n\n".join(doc.page_content for doc in docs)

    def invoke(self, query):
        return self.rag_chain.invoke(query)

# # 사용 예:
# pipeline = RAGPipeline()
# result = pipeline.invoke("질문 내용")

""" ref
https://python.langchain.com/docs/use_cases/question_answering/vector_db_qa
https://js.langchain.com/docs/modules/chains/popular/vector_db_qa/
https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa
"""

#setup chain
# chain = RetrievalQA.from_chain_type(
#                     llm=llm,
#                     chain_type="stuff",
#                     retriever=ensemble_retriever,
#                     )

##### 예시
# response = chain.run("당신은 한국의 복지 전문가입니다. 주어진 정보만을 가지고, 다음의 질문에 대답하면 됩니다. 질문 : 농촌 풍수해 피해의 경우 보상받을 수 있는 방법은? 답변 : ...")
# print(response)
