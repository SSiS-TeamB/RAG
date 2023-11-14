from chromaClient import ChromaClient

chroma_client = ChromaClient()
print(chroma_client.collection_list)

from chromadb.utils import embedding_functions

base_model = "BM-K/KoSimCSE-roberta-multitask"
emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=base_model, normalize_embeddings=True)

query = "졸라 배고파ㅏㅏㅏㅏㅏ"

# 크로마디비로 직접 서치
chroma_client.connect_collection('wf_schema', emb_func=emb_func)
chroma_client.collection.count()
ans1 = chroma_client.semantic_search(['배고파ㅏㅏㅏㅏ'], 3)


# langchain 으로 서치
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
import torch

emb_info_dict = {'model_name': base_model, 'model_kwargs': {'device': "cuda" if torch.cuda.is_available() else "cpu"},
    'encode_kwargs': {'normalize_embeddings': True}}
lc_emb_func = SentenceTransformerEmbeddings(**emb_info_dict)

vector_store = Chroma(collection_name='wf_schema', persist_directory='./chroma_storage', embedding_function=lc_emb_func)
rtr = vector_store.as_retriever(search_type='mmr')

ans2 = vector_store.max_marginal_relevance_search(query)
ans3 = rtr.get_relevant_documents(query)

print(ans1)
print('*'*50)
print(ans2)
print('*'*50)
print(ans3)

# from langchain.chains import RetrievalQA
# from langchain.chains import HypotheticalDocumentEmbedder
# from langchain.chains import LLMChain

# from langchain.chat_models import ChatOpenAI

# from langchain.vectorstores.chroma import Chroma

# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# from langchain.prompts import PromptTemplate

# from langchain.retrievers import BM25Retriever, EnsembleRetriever

# # llm: OpenAI
# llm = ""

# # setup chain
# chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', )

# vectorstore = Chroma(collection_name="vector_db", persist_directory="./chroma_storage", embedding_function=HyDEembeddings)




