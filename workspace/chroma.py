import os
from document import BaseDBLoader

from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

""" set up vector DB in local path. (./chroma) """

#directory settings
directory = os.path.dirname(__file__)
os.chdir(directory)

#embedding config (edit later if domain-adaptation complete.)

def _device_check() : 
    ''' for check cuda availability '''
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

embedding = SentenceTransformerEmbeddings(
    model_name="BM-K/KoSimCSE-roberta-multitask", 
    model_kwargs={'device':_device_check()}, 
    encode_kwargs={'normalize_embeddings':True},
    )

#dbsetup
result_storage = BaseDBLoader().load()

## chroma setting w.langchain (no parentretriever)
vectorstore = Chroma.from_documents(result_storage, embedding, persist_directory="./chroma")
vectorstore.persist()
print("There are", vectorstore._collection.count(), "in the collection.")
