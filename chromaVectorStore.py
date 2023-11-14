from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings as STE

from workspace.mdLoader import BaseDBLoader


class ChromaVectorStore:
    def __init__(self, **kwargs) -> None:
        
        self.vs = Chroma(**kwargs)
        self.retriever = self.vs.as_retriever(search_type='mmr')
        return
    
    def retrieve(self, query):
        ans1 = self.retriever.get_relevant_documents(query)
        ans2 = self.vs.max_marginal_relevance_search(query)
        return [ans1, ans2]

    def load_docs(self, emb:STE, dir_path:str):
        doc_loader = BaseDBLoader()
        docs = doc_loader.load()
        vectorstore = Chroma.from_documents(docs, emb, dir_path)
        vectorstore.persist()
        print("There are", vectorstore._collection.count(), "in the collection.")
        return