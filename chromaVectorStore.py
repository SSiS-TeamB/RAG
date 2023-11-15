from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings as STE

from workspace.mdLoader import BaseDBLoader
from workspace.embeddingSetup import EmbeddingLoader


class ChromaVectorStore:
    def __init__(self, **kwargs) -> None:
        self.emb = EmbeddingLoader().load()
        kwargs['embedding_function'] = self.emb

        self.vs = Chroma(**kwargs)
        self.retriever = self.vs.as_retriever(search_type='mmr')
        return
    
    def retrieve(self, emb:STE, query:str, is_sim_search=False) -> list:
        """ VectorDB에 query(질문) 넣어서 나오는 답변 찾아오기. 
            *args
            query : string 형태로 VectorDB에 유사도 기반 검색할 때 사용하는 질문
            is_sim_search : search 방식 선택(기본 mmr, True로 하면 similarity_search)
        """
        ### query - vector 변환
        query = self.emb.embed_query(query)

        ### 여기에 self.embedding_model로 query vector로 변환하는거 넣어라
        if is_sim_search:
            answer = self.vs.similarity_search_by_vector(query)
        else :
            answer = self.vs.max_marginal_relevance_search_by_vector(query)
        return answer

    def load_docs(self, dir_path:str):
        doc_loader = BaseDBLoader()
        docs = doc_loader.load()
        vectorstore = Chroma.from_documents(docs, self.emb, dir_path)
        vectorstore.persist()
        print("There are", vectorstore._collection.count(), "in the collection.")
        return

