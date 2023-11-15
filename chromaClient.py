import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction as STEF
from chromadb.config import Settings

from workspace.mdLoader import BaseDBLoader
import uuid


class ChromaClient:
    def __init__(self, store_dir="chroma_storage") -> None:
        self.client = chromadb.PersistentClient(path=store_dir, settings=Settings(allow_reset=True))
        self.collection_list = self.client.list_collections()
        self.collection = None
        return

    # $$$ get_or_create_collection 로 바꿔도 무방
    def make_collection(self, coll_name, emb_model:str, meta_dict={"hnsw:space": "cosine"}):
        if emb_func:
            emb_func = STEF(model_name=emb_model, normalize_embeddings=True)
            self.client.create_collection(name=coll_name, embedding_function=emb_func, metadata=meta_dict)
        else:
            self.client.create_collection(name=coll_name, metadata=meta_dict)
        return
    def connect_collection(self, coll_name, emb_model:str):
        if emb_model:
            emb_func = STEF(model_name=emb_model, normalize_embeddings=True)
            self.collection = self.client.get_collection(name=coll_name, embedding_function=emb_func)
        else:
            self.client.get_collection(name=coll_name)
        return
    
    def load_doc_and_add_data(self, path_db:str):
        doc_loader = BaseDBLoader(path_db=path_db)
        docs = doc_loader.load()

        # $$$ 이 부분 get_corpus() 사용하는 걸로 수정하면 되나?
        idx = 0
        for doc in docs:
            idx += 1
            uuid_name = uuid.uuid1()
            print(f"{idx}) document_uuid: {uuid_name}")
            self.collection.add(ids=[str(uuid_name)], documents=doc.page_content)
        return
    # def read_data(self):
    #     return
    
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