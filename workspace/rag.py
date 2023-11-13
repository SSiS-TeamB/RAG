import os

from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

#api key(추가해서 쓰시오)
import settings

from langchain.llms import OpenAI, GooglePalm
from langchain.chains import RetrievalQA

""" HyDE 붙이기 + """

# path setup
directory = os.path.dirname(__file__)
os.chdir(directory)


# embedding config (edit later if domain-adaptation complete.)
def _device_check():
    ''' for check cuda availability '''
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # torch.backends.mps.is_available()
    return device


embedding = SentenceTransformerEmbeddings(
    model_name="BM-K/KoSimCSE-roberta-multitask", 
    model_kwargs={'device': _device_check()},
    encode_kwargs={'normalize_embeddings': True},
    )

# get chroma collection from ./chroma
vector_store = Chroma(persist_directory="./chroma", embedding_function=embedding)

# get llm

############# 일단 OpenAI 오류로 안됨 돌리지마라
os.environ["OPENAI_API_KEY"] = settings.openai_api_key
# llm = GooglePalm(google_api_key=settings.PALM_api_key, temperature=0, max_output_tokens=512)
llm = OpenAI(temperature=1)
# $$$ temperature 가 뭐임?

# retrieve 방식에 따른 차이 필요(Ensemble, search type 수정 등등)
# setup chain
chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vector_store.as_retriever(search_type='mmr'),
                )

"""
https://python.langchain.com/docs/use_cases/question_answering/vector_db_qa
https://js.langchain.com/docs/modules/chains/popular/vector_db_qa/
https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa
"""

#### prompt 수정 필요
result = chain.run("40대 주거 욕구에 대해 말해줘.")
print(result)
