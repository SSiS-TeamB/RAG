import torch
import chromadb

from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings


# def _device_check() : 
#     return "cuda" if torch.cuda.is_available() else "cpu"


# embedding = SentenceTransformerEmbeddings(
#     model_name="BM-K/KoSimCSE-roberta-multitask", 
#     model_kwargs={'device': _device_check()},
#     encode_kwargs={'normalize_embeddings': True},
#     )

# vector_store = Chroma(persist_directory="./chroma", embedding_function=embedding)
# print(1, vector_store)


# vec_store = vector_store.afrom_documents()

# docs = vector_store.asimilarity_search(q)
# print(docs)

chroma_client = chromadb.PersistentClient()
coll_list = chroma_client.list_collections()
print(coll_list)
# chroma_client.delete_collection(name='langchain')

# coll1 = chroma_client.get_collection("langchain")
# print(coll1)
# print(coll1.get())
# print(coll1.peek())
# print(coll1.count())

q = '국민 취업 준비 20대'