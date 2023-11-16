from langchain.vectorstores.chroma import Chroma

import pickle
from langchain.schema.document import Document

from workspace.embeddingSetup import EmbeddingLoader
from workspace.mdLoader import BaseDBLoader

import os

class ChromaVectorStore:
    def __init__(self, **kwargs) -> None:
        self.emb = EmbeddingLoader().load()
        kwargs['embedding_function'] = self.emb
        self.vs = Chroma(**kwargs)
        self.vs_dir_path = kwargs['persist_directory']
        self.vs_coll_name = kwargs['collection_name']

        self.retriever = self.vs.as_retriever(search_type='mmr')
        
        return
    
    def retrieve(self, query:str, is_sim_search=False) -> list:
        """ VectorDB에 query(질문) 넣어서 나오는 답변 찾아오기.
            *args
            query : string 형태로 VectorDB에 유사도 기반 검색할 때 사용하는 질문
            is_sim_search : search 방식 선택(기본 mmr, True로 하면 similarity_search)
        """
        ### query - vector 변환
        print(query)
        query = self.emb.embed_query(query)
        
        ### search
        if is_sim_search:
            answer = self.vs.similarity_search_by_vector(query)
        else :
            answer = self.vs.max_marginal_relevance_search_by_vector(query)

        return answer
    
    def _get_pickle(self, documents:list[Document], save_path:str) -> None:
        """ pickle file(for BM25 documents) -> 빠른 loading 위해서 file 형식으로 저장 """
        root_path_split = save_path.split("/")
        result_save_path = os.path.join(root_path_split[0], 'document.pkl')
        result_save_path = result_save_path.replace("\\","/")
        with open(result_save_path, 'wb') as file :
            pickle.dump(documents, file)
        print(f'pickle file use for BM25 has been saved to path : {result_save_path}')

        return

    def load_docs(self, document_path:str, is_split=True, is_regex=True) -> None :
        doc_loader = BaseDBLoader(document_path)
        docs = doc_loader.load(is_split, is_regex)
        self._get_pickle(documents=docs, save_path=document_path) #get pickle file -> to Save time, save list of Documents

        vectorstore = Chroma.from_documents(documents=docs, embedding=self.emb, collection_name=self.vs_coll_name, persist_directory=self.vs_dir_path)
        vectorstore.persist()
        print("There are", vectorstore._collection.count(), "in the collection.")

        return

## 사용예제
if __name__ == "__main__":
    vectorstore = ChromaVectorStore(**{
        "collection_name":"wf_schema_no_split",
        "persist_directory":"workspace/chroma_storage",
    })

    vectorstore.load_docs(document_path="workspace/markdownDB", is_regex=True, is_split=False)
