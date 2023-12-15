from workspace.mdLoader import BaseDBLoader
from chromaVectorStore import ChromaVectorStore, EnsembleRetrieverWithFilter
from chromaClient import ChromaClient


# test_loader = BaseDBLoader()
# docs = test_loader.load()
# print(len(docs))

query_text = '대학생인데 해외 유학 가고 싶다'

# base_model = "BM-K/KoSimCSE-roberta-multitask"
base_model = "da_finetune_epoch_2"
vs_info_dict = {'model_name': base_model, 'collection_name': 'langchain', 'persist_directory': 'workspace/chroma_storage'}
vector_store = EnsembleRetrieverWithFilter(**vs_info_dict)


# res = vector_store.retrieve(query_text)
vector_store.ensemble_retriever.retrievers[1].search_kwargs['filter'] = {'source': 'workspace\\markdownDB\\04_청소년•청년_지원\\04_청소년_국제교류.md'}
res2 = vector_store.retriever.get_relevant_documents(query_text)
# res = vector_store.retriever.get_relevant_documents(query_text, filter={'source': 'workspace\\markdownDB\\02_취업_지원\\02_여성특화제품_해외_진출_지원.md'})


print('-'*30)
for r in res2:
    print(r.metadata)

# cc = ChromaClient()
# print(cc.collection_list)

