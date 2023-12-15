import torch
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings as STE
import re
import pickle

from langchain.retrievers import BM25Retriever, EnsembleRetriever


from workspace.mdLoader import BaseDBLoader



class ChromaVectorStore:
    def __init__(self, **kwargs) -> None:
        emb_info_dict = {'model_name': kwargs.pop('model_name'), 'model_kwargs': {'device': "cuda" if torch.cuda.is_available() else "cpu"},
'encode_kwargs': {'normalize_embeddings': True}}
        self.emb = STE(**emb_info_dict)
        kwargs['embedding_function'] = self.emb

        self.vs = Chroma(**kwargs)
        self.retriever = self.vs.as_retriever(search_type='mmr', search_kwargs={'k':5})
        return
    
    def retrieve(self, query):
        ans1 = self.retriever.get_relevant_documents(query)
        ans2 = self.vs.max_marginal_relevance_search(query, k=10)
        # ans2 = self.vs.max_marginal_relevance_search(query, k=10, where_document={'$contains': '대한민국'})
        # ans2 = self.vs.max_marginal_relevance_search(query, k=10, filter={'source': {"$eq": 'markdowndb\\04_청소년•청년_지원\\04_청소년_국제교류.md'}})
        # $gt, $gte, $lt, $lte, $ne, $eq, $in, $nin
        # where: {"$and": ["name": {"$eq": "John Doe"}, "age": {"$gte": 30}]}
        return [ans1, ans2]

    def load_docs(self, per_dir_path:str, model_name:str = "BM-K/KoSimCSE-roberta-multitask"):
        emb_info_dict = {'model_name': model_name, 'model_kwargs': {'device': "cuda" if torch.cuda.is_available() else "cpu"},
'encode_kwargs': {'normalize_embeddings': True}}
        emb = STE(**emb_info_dict)

        doc_loader = BaseDBLoader()
        docs = doc_loader.load()
        vectorstore = Chroma.from_documents(docs, emb, per_dir_path)
        vectorstore.persist()
        print("There are", vectorstore._collection.count(), "in the collection.")
        return

    @staticmethod
    def show_full(docs):
        # sep_str = f"\n{'-'*50}\n{'-'*50}\n"
        sep_str = f"\n\n"
        result = []

        cnt = 1
        for doc in docs:
            metadata = doc.metadata
            title_resource = str(metadata).split(":")[1].lstrip()
            title_resource = re.sub(pattern="}|'", repl="", string=title_resource)
            title_resource = title_resource.split("\\")[4]+"_"+title_resource.split("\\")[-1]

            content = doc.page_content
            formatted_document = f"<<{cnt}>> {title_resource}\n({len(content)})  {content} \n metadata: {metadata}"
            result.append(formatted_document)
            cnt += 1

        return sep_str.join(doc for doc in result)

    @staticmethod
    def format_docs(docs):
        ## 어느 제도 부분에서 가져왔는지 나타내는 출처 : medata 활용해서 같이 출력
        sep_str = "\n\n"
        result = []

        for doc in docs:
            ### 이 부분도 수정해야 함.. (key : value로)
            metadata = doc.metadata
            title_resource = str(metadata).split(":")[1].lstrip()
            title_resource = re.sub(pattern="}|'", repl="", string=title_resource)
            title_resource = title_resource.split("\\")[4]+"_"+title_resource.split("\\")[-1]
            
            content = doc.page_content
            content_splitted = content.split('\n\n')
            title = content_splitted[0]

            displayed_text = " ".join(content_splitted[1:])[:300]
            displayed_text = re.sub('\n+', ' ', displayed_text)
            # if len(displayed_text) > 300:
            #     displayed_text = displayed_text[297]+' ...'
            displayed_text += " ..."

            content = f"[{title}]\n\n {displayed_text}"
            formatted_document = content + f"\n\n 출처 : {title_resource}"
            result.append(formatted_document)
        
        return sep_str.join(doc for doc in result)


class EnsembleRetrieverWithFilter(ChromaVectorStore):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        with open('workspace/document.pkl', 'rb') as file:
            self.documents = pickle.load(file)

        self.bm25_retriever = BM25Retriever.from_documents(documents=self.documents)
        self.bm25_retriever.k = 1
        self.ensemble_retriever = EnsembleRetriever(retrievers=[self.bm25_retriever, self.retriever], weights=[0.1, 0.9])