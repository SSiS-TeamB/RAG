import os
import chromadb
from langchain.vectorstores import chroma

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


""" set up vector DB in local path. """

def _device_check() : 
    ''' for check cuda availability '''
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

#directory settings
directory = os.path.dirname(__file__)
os.chdir(directory)

#embedding config (edit later if domain-adaptation complete.)
embedding = SentenceTransformerEmbeddings(
    model_name="BM-K/KoSimCSE-roberta", 
    model_kwargs={'device':_device_check()}, 
    encode_kwargs={'normalize_embeddings':True},
    )

#chromadb config
collection_name = "vector_db"

vectorstore = Chroma(
    client=chromadb.PersistentClient(),
    collection_name = collection_name,
    embedding_function = embedding,
    persist_directory="./chroma",
)

#textsplitter config
text_splitter=RecursiveCharacterTextSplitter(
    separators="\n\n",
    chunk_size=200,
    is_separator_regex=False,
)

#dbsetup
list_of_path_db = os.listdir("./markdowndb")

result_storage = []
######## 여기서부터 수정 ㄱㄱ
for db_folder in list_of_path_db:
    directory_loader = DirectoryLoader(path=os.path.join("./markdowndb", db_folder), loader_cls=UnstructuredMarkdownLoader)
    result=directory_loader.load_and_split(text_splitter=text_splitter)

vectorstore.from_documents(result_storage)
vectorstore.persist()
