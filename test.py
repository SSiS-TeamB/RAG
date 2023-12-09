from workspace.mdLoader import BaseDBLoader
from chromaVectorStore import ChromaVectorStore
from chromaClient import ChromaClient


# test_loader = BaseDBLoader()
# docs = test_loader.load()
# print(len(docs))

query_text = '대학생인데 해외 유학 가고 싶다'

# base_model = "BM-K/KoSimCSE-roberta-multitask"
base_model = "da_finetune_epoch_2"
vs_info_dict = {'model_name': base_model, 'collection_name': 'langchain', 'persist_directory': 'workspace/chroma_storage'}
vector_store = ChromaVectorStore(**vs_info_dict)

# res = vector_store.retrieve(query_text)
res = vector_store.vs.get()

print(len(res))

# cc = ChromaClient()
# print(cc.collection_list)

