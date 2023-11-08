import os

from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

#api key(추가해서 쓰시오)
import settings

## OpenAI 모델을 사용하기 위한 Class 
from langchain.llms import OpenAI

## 다른 LLM을 호출하기 위해 chains 사용
## 우리는 openAI chatbot - vectorstore 연결을 위해 사용
## 외부 DB를 참고하여 QA를 해주는 거기 떄문에 RetrievalQA 사용
from langchain.chains import RetrievalQA

""" HyDE 붙이기 + """

#path setup
# 디렉토리 설정
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

#get chroma collection from ./chroma
# 만들어져 있는 ./chroma  --> vector DB 가져오기
vectorstore = Chroma(persist_directory="./chroma", embedding_function=embedding)

#get llm

############# 일단 OpenAI 오류로 안됨 돌리지마라
## opeanAI API 사용위한 파이썬 환경설정
os.environ["OPENAI_API_KEY"] = settings.openai_api_key
# llm = GooglePalm(google_api_key=settings.PALM_api_key, temperature=0, max_output_tokens=512)
llm = OpenAI(temperature=1) #temperature(0~1) : 창의성 ( 0 : 일관된 답 , 1 : 창의력  max)

#retrieve 방식에 따른 차이 필요(Ensemble, search type 수정 등등)
#setup chain
## 연결할 chain방법을 config(설정)
chain = RetrievalQA.from_chain_type(
                    llm=llm, # 연결할 chatbot model
                    chain_type="stuff",  # uses ALL of the text from the documents in the prompt
                                         # 총 4가지 종류
                    retriever=vectorstore.as_retriever(search_type='mmr'), # vectorstore 를  retriever로 사용
                                                                        # search_type 은 관련성 높은 문서를 찾는 방법
                                                                # 2가지 종류 : similariry = 코사인 유사도
                                                                #              mmr = Maximum Marginal Relevance
                )

####prompt 수정 필요
## component 들을 연결한 chain을 실행
## chatbot 이니까  답변 줌
result = chain.run("40대 주거 욕구에 대해 말해줘.")
print(result)