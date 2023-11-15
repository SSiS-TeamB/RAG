from chromaVectorStore import ChromaVectorStore
from chromaClient import ChromaClient

base_model = "BM-K/KoSimCSE-roberta-multitask"

q = '졸라 배고파ㅏㅏㅏㅏㅏㅏ'

# Settings for semantic_search using "chromadb" module
chroma_client = ChromaClient()
chroma_client.connect_collection('wf_schema', base_model)
res = chroma_client.semantic_search([q])

print(res)