import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction as SEF
from chromadb.config import Settings


class ChromaClient:

    def __init__(self, store_dir="chroma_storage") -> None:
        self.client = chromadb.PersistentClient(path=store_dir, settings=Settings(allow_reset=True))

        self.collections = self.client.list_collections()
        self.collection = None
        return

    # $$$ get_or_create_collection 로 바꿔도 무방
    def make_collection(self, coll_name, emb_func:SEF, meta_dict={"hnsw:space": "cosine"}):
        if emb_func:
            self.client.create_collection(name=coll_name, embedding_function=emb_func, metadata=meta_dict)
        else:
            self.client.create_collection(name=coll_name, metadata=meta_dict)
        return
    def connect_collection(self, coll_name, emb_func:SEF):
        if emb_func:
            self.collection = self.client.get_collection(name=coll_name, embedding_function=emb_func)
        else:
            self.client.get_collection(name=coll_name)
        return
    
    # $$$ 아래 기능은 나중에 하자
    def add_data(self):
        return
    def read_data(self):
        return
    
    # Semantic Search
    def semantic_search(self, q_list:list, k:int = 5) -> dict:
        return self.collection.query(query_texts=q_list, n_results=k)

    # $$$ 리셋하면 다 지워지는 건지, 확인 후 수정하기!
    def reset_client(self):
        self.client.reset()
        return
    pass


"""
참조 링크
https://webcache.googleusercontent.com/search?q=cache:https://medium.com/@kbdhunga/an-overview-of-chromadb-the-vector-database-206437541bdd

"""
