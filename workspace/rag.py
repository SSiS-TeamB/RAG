import os

from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

#api key(추가해서 쓰시오)
import settings

from langchain.llms.openai import OpenAI
from langchain.llms.google_palm import GooglePalm
from langchain.chains import RetrievalQA

""" HyDE 붙이기 + """

#path setup
directory = os.path.dirname(__file__)
os.chdir(directory)

#embedding config (edit later if domain-adaptation complete.)
def _device_check() : 
    ''' for check cuda availability '''
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # torch.backends.mps.is_available()
    return device

embedding = SentenceTransformerEmbeddings(
    model_name="BM-K/KoSimCSE-roberta-multitask", 
    model_kwargs={'device':_device_check()}, 
    encode_kwargs={'normalize_embeddings':True},
    )

#get chroma collection from ./chroma
vectorstore = Chroma(persist_directory="./chroma", embedding_function=embedding)

#get llm

############# 일단 OpenAI 오류로 안됨 돌리지마라 #############

os.environ["OPENAI_API_KEY"] = settings.openai_api_key
# llm = GooglePalm(google_api_key=settings.PALM_api_key, temperature=0, max_output_tokens=512)
llm = OpenAI(temperature=1)

#retrieve 방식에 따른 차이 필요(Ensemble, search type 수정 등등)
#setup chain
chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_type='mmr'),
                )

####prompt 수정 필요
result = chain.run("문서를 기반으로 했을 때 탈북자 지원금에 대해 설명해줘.")
print(result)

############# 일단 OpenAI 오류로 안됨 돌리지마라 #############