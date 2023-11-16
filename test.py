from workspace.mdLoader import BaseDBLoader
from chromaVectorStore import ChromaVectorStore
from chromaClient import ChromaClient


# test_loader = BaseDBLoader()
# docs = test_loader.load()
# print(len(docs))

query_text = '배고파ㅏㅏㅏ'

base_model = "BM-K/KoSimCSE-roberta-multitask"
vs_info_dict = {'model_name': base_model, 'collection_name': 'langchain', 'persist_directory': 'workspace/chroma_storage'}
vector_store = ChromaVectorStore(**vs_info_dict)

res = vector_store.retrieve(query_text)

print(res)

# cc = ChromaClient()
# print(cc.collection_list)

