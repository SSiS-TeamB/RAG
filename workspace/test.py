from langchain.chains import RetrievalQA
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.chains import LLMChain

from langchain.chat_models import ChatOpenAI

from langchain.vectorstores.chroma import Chroma

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain.prompts import PromptTemplate

from langchain.retrievers import BM25Retriever, EnsembleRetriever

# llm: OpenAI
llm = ""

# setup chain
chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', )

vectorstore = Chroma(collection_name="vector_db", persist_directory="./chroma_storage", embedding_function=HyDEembeddings)