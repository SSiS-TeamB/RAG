import os
from document import BaseDBLoader

from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

import pickle

""" set up vector DB in local path. (./chroma_storage) """

#directory settings
directory = os.path.dirname(__file__)
os.chdir(directory)

def _device_check() : 
    ''' for check cuda availability '''
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

#embedding config
embedding = SentenceTransformerEmbeddings(
    model_name="da_finetune_epoch_2", 
    model_kwargs={'device':_device_check()}, 
    encode_kwargs={'normalize_embeddings':True},
    )

#dbsetup
result_storage = BaseDBLoader().load(is_split=False, is_regex=True)

## pickle로 Document 따로 저장(일단 수정 안하고 임시로 저장함)
with open('./document.pkl', 'wb') as file :
    pickle.dump(result_storage, file)

## chroma setting w.langchain (no parentretriever)
vectorstore = Chroma.from_documents(result_storage, embedding, persist_directory="./chroma_storage", collection_name="wf_schema")
vectorstore.persist()
print("There are", vectorstore._collection.count(), "in the collection.")


