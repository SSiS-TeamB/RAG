import re
import os
import pickle
import math
# api key(추가해서 쓰시오)
from workspace.settings import OPENAI_API_KEY
from workspace.analogicalPrompt import generateAnalogicalPrompt, get_normal_prompt

from langchain.chat_models import ChatOpenAI
from langchain.retrievers import BM25Retriever, EnsembleRetriever

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser

from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore

from chromaVectorStore import ChromaVectorStore
from workspace.mdLoader import BaseDBLoader 
from datetime import datetime

from transformers import AutoTokenizer

#api key settings
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

class RAGPipeline:
    def __init__(self, model, vectorstore:Chroma, embedding, filter_dict):
        # print("RAGPipeline 새로 만들고있음!!!!!!")
        
        self.llm = ChatOpenAI(model=model, temperature=0.1, streaming=True)
        self.vectorstore = vectorstore
        self.embedding = embedding
        self.filter_dict = filter_dict

        self.tokenizer_base = AutoTokenizer.from_pretrained("workspace/model/jhgan_seed_777_lr_1e-5_final")
        
        save_path = "workspace/document.pkl"
        if not os.path.exists(save_path) :
            document = BaseDBLoader("workspace/markdownDB").load(is_split=False, is_regex=True)
            ChromaVectorStore.get_pickle(documents=document, save_path=save_path)

        # pickle list 객체 생성 시 로드
        with open('workspace/document.pkl', 'rb') as file:
            self.documents = pickle.load(file)

        child_splitter = SentenceTransformersTokenTextSplitter(
            tokens_per_chunk=128,
            model_name="workspace/model/jhgan_seed_777_lr_1e-5_final",
            # model_name="workspace/model/dadt_epoch2_kha_tok",
            chunk_overlap=10,
        )

        #### encode cachefile into byte(to use ParentDocumentRetriever)
        fs = LocalFileStore("cache")
        store = create_kv_docstore(fs)

        # ParentDocumentRetriever
        self.parent_retreiver = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            search_type="similarity_score_threshold",   
            # search_kwargs={"k":7},
            search_kwargs={"score_threshold":0.5, "k":7},
        )

        ## check cachefile exsists 
        if not list(store.yield_keys()) :
            dbloader = BaseDBLoader(path_db="workspace/markdownDB/")
            self.parent_retreiver.add_documents(dbloader.load(is_split=False, is_regex=True))
        
        # BM25 Retriever
        if self.filter_dict:
            if "category" in self.filter_dict:
                filtered_docs = [d for d in self.documents if d.metadata["category"] == self.filter_dict["category"]["$eq"]]
            else:
                filtered_docs = [d for d in self.documents if d.metadata["category"] in [e['category']['$eq'] for e in self.filter_dict['$or']]]
            self.bm25_retriever = BM25Retriever.from_documents(documents=filtered_docs, preprocess_func=self.bm_parse)
        else:
            self.bm25_retriever = BM25Retriever.from_documents(documents=self.documents, preprocess_func=self.bm_parse)
        self.bm25_retriever.k = 2

        # self.bm25_retriever = BM25Retriever.from_documents(documents=self.documents, preprocess_func=self.bm_parse)

        # Ensemble
        self.ensemble_retriever = EnsembleRetriever(retrievers=[self.bm25_retriever, self.parent_retreiver], weights=[0.1, 0.9])
        # RAG Chain
        self.rag_chain = (
            {"context": self.ensemble_retriever | self.format_docs, "question": RunnablePassthrough()}
            | get_normal_prompt()
            | self.llm
            | StrOutputParser()
        )

    # $$$ BM25 parsing
    def bm_parse(self, text:str) -> list[str]:
        # tokenizer_base = AutoTokenizer.from_pretrained("workspace/model/dadt_epoch2_kha_tok")
        splitted_text = text.split()
        
        tokenized_list = []
        # 300 토큰씩
        splt_tok_num = 100
        docs_num = math.ceil(len(splitted_text) / splt_tok_num)
        for i in range(docs_num):
            tokenized_list.extend([tok.replace("##", "") for tok in self.tokenizer_base.tokenize(" ".join(splitted_text[splt_tok_num*i:splt_tok_num*(i+1)]))])
        return tokenized_list

    @staticmethod
    def format_docs(docs):
        ## 어느 제도 부분에서 가져왔는지 나타내는 출처 : medata 활용해서 같이 출력
        sep_str = "\n***\n"
        result = []

        for doc in docs:
            ### 이 부분도 수정해야 함.. (key : value로)
            content = doc.page_content 
            content_splitted = content.split('\n\n')

            displayed_text = " ".join(content_splitted[:])[:]
            displayed_text = re.sub('\n+', ' ', displayed_text)
            displayed_text += " ..."
            
            
            metadata = doc.metadata
            if metadata.get('url') is not None:
                url = f"""<a href="{metadata['url']}">{metadata['title']}</a>"""
            else:
                url = metadata['title']

            # unsafe_allow_html=True,
            formatted_document = f"[{url}]\n\n> 내용\n\n$$${displayed_text}$$$\n\n> 카테고리\n\n {metadata['tag']}\n\n> 주제\n\n{metadata['category']}\n\n"
            #### metadata 붙인거 추가하기.......
            result.append(formatted_document)
        
        return sep_str.join(doc for doc in result)
    # @staticmethod
    # def format_docs(docs):
    #     ## 어느 제도 부분에서 가져왔는지 나타내는 출처 : medata 활용해서 같이 출력
    #     sep_str = "\n\n"
    #     result = []

    #     for doc in docs:
    #         ### 이 부분도 수정해야 함.. (key : value로)
    #         metadata = doc.metadata
    #         content = doc.page_content 
    #         content_splitted = content.split('\n\n')

    #         displayed_text = " ".join(content_splitted[1:])[:300]
    #         displayed_text = re.sub('\n+', ' ', displayed_text)
    #         displayed_text += " ..."
            
            
    #         if metadata.get('url') is not None :
    #             url = f"""<a href="{metadata['url']}">{metadata['title']}</a>"""
    #         else :
    #             url = metadata['title']

    #         # unsafe_allow_html=True,
    #         content = f"[{url}]\n\n> 내용 \n\n{displayed_text}"
    #         formatted_document = content + f"\n\n> 카테고리\n\n {metadata['tag']} \n\n 주제: {metadata['category']}"
    #         #### metadata 붙인거 추가하기.......
    #         result.append(formatted_document)
        
    #     return sep_str.join(doc for doc in result)

    def invoke(self, query):
        result = self.rag_chain.invoke(query)
        return result
    
    def retrieve(self, query):
        # query = self.embedding.embed_query(query)
        self.ensemble_retriever.retrievers[1].search_kwargs['filter'] = self.filter_dict
        # result = self.ensemble_retriever.retrievers[1].get_relevant_documents(query)
        result = self.ensemble_retriever.get_relevant_documents(query)
        return result
""" ref
https://python.langchain.com/docs/use_cases/question_answering/vector_db_qa
https://js.langchain.com/docs/modules/chains/popular/vector_db_qa/
https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa
"""

# 사용 예:
if __name__ == "__main__":
    start_time = datetime.now()

    collection_name = "wf_schema_split"
    persist_directory = "workspace/chroma_storage"

    vectorstore = ChromaVectorStore(**{
        "collection_name":collection_name, 
        "persist_directory":persist_directory,
        "collection_metadata" : {"hnsw:space":"cosine"}
    })

    model = "gpt-3.5-turbo-1106"
    # model = "gpt-4-1106-preview"

    filter_dict={'category': {"$eq": '10 기타 위기별·상황별 지원'}}

    rag_pipeline = RAGPipeline(vectorstore=vectorstore.vs, embedding=vectorstore.emb, model=model, filter_dict=filter_dict)

    retrieval_result = rag_pipeline.retrieve("주택 공급")
    # print(retrieval_result)
    # print(len(retrieval_result))
    end_time = datetime.now()
    # print((end_time-start_time).total_seconds(),"seconds.") ### timecheck 11-26:

