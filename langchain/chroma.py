import os
import chromadb
from langchain.vectorstores import Chroma

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


""" set up vector DB in local path. (./chroma) """

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
    model_name="BM-K/KoSimCSE-roberta-multitask", 
    model_kwargs={'device':_device_check()}, 
    encode_kwargs={'normalize_embeddings':True},
    )

#chromadb config
# persistent_client = chromadb.PersistentClient()
# collection_name = "vector_db"

# collection = persistent_client.get_or_create_collection(collection_name)

# vectorstore = Chroma(
#     client=persistent_client,
#     collection_name = collection_name,
#     embedding_function = embedding,
#     persist_directory="./chroma",
# )

# print("There are", vectorstore._collection.count(), "in the collection.")


#textsplitter config
text_splitter=RecursiveCharacterTextSplitter(
    separators="\n\n",
    chunk_size=200,
    is_separator_regex=False,
)

#dbsetup
list_of_path_db = os.listdir("./markdowndb")

result_storage = []


### 여기 document.py 만들면 통일 ㄱㄱ
for db_folder in list_of_path_db:
    directory_loader = DirectoryLoader(path=os.path.join("./markdowndb", db_folder), loader_cls=UnstructuredMarkdownLoader)
    result=directory_loader.load_and_split(text_splitter=text_splitter)
    result_storage.extend(result)


## chroma setting w.langchain (no parentretriever)
vectorstore = Chroma.from_documents(result_storage, embedding, persist_directory="./chroma")
vectorstore.persist()

print("There are", vectorstore._collection.count(), "in the collection.")
