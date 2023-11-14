from langchain.vectorstores import Chroma


class ChromaVectorStore:
    def __init__(self, **kwargs) -> None:
        
        self.vs = Chroma(**kwargs)
        self.retriever = self.vs.as_retriever(search_type='mmr')
        return
    
    def retrieve(self, query):
        ans1 = self.retriever.get_relevant_documents(query)
        ans2 = self.vs.max_marginal_relevance_search(query)
        return [ans1, ans2]

