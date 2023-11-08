import os
from document import BaseDBLoader

from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

""" set up vector DB in local path. (./chroma) """

#directory settings
#실행할 디렉토리 변경
directory = os.path.dirname(__file__)
os.chdir(directory)

#embedding config (edit later if domain-adaptation complete.)
#Embedding 환경설정 및 모델 연결
def _device_check() : 
    ''' for check cuda availability '''
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


## Embedding 환경설정
embedding = SentenceTransformerEmbeddings(
    model_name="BM-K/KoSimCSE-roberta-multitask",   # 사용할 모델명
    model_kwargs={'device':_device_check()},        # 모델에 전달할 파라미터들
    encode_kwargs={'normalize_embeddings':True},    # 인코딩시 필요한 파라미터 설정
    )

#dbsetup
# document.py 의 BaseDBLoader class를 통해 markdownDB를 splitting 해서 저장
result_storage = BaseDBLoader().load()

## chroma setting w.langchain (no parentretriever)
## result_storage(text)를 Embedding하여  Chroma DB 생성후 local Disk 에 저장 -> 초기화 및 재사용 용이
## Chroma.from_documents(vector화 할 documnet_list, Embedding_config, In-memory directory)
## ChrmaDB객체.persist() -> In-Memory 에 있는 Vector DB를 Disk로 이동
vectorstore = Chroma.from_documents(result_storage, embedding, persist_directory="./chroma")
vectorstore.persist()
print("There are", vectorstore._collection.count(), "in the collection.")
## _collection : vectorstore에 저장된 collection에 접근하는 메소드?
## _collection.count() : vectorstore에 저장된 collection의 개수 반환