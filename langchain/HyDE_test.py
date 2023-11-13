import os

from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

#api key(추가해서 쓰시오)
import settings

from langchain.llms import OpenAI, GooglePalm
from langchain.chains import LLMChain, RetrievalQA, HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate

""" HyDE 붙이기 + """

#path setup
directory = os.path.dirname(__file__)
os.chdir(directory)

#embedding config (edit later if domain-adaptation complete.)
def _device_check() : 
    def _device_check():
        ''' for check cuda availability '''
        if torch.cuda.is_available() : device ="cuda"
        elif torch.backends.mps.is_available() : device = "mps"
        else : device = "cpu"
        return device

embedding = SentenceTransformerEmbeddings(
    model_name="BM-K/KoSimCSE-roberta-multitask", 
    model_kwargs={'device':_device_check()}, 
    encode_kwargs={'normalize_embeddings':True},
    )

#get chroma collection from ./chroma
vectorstore = Chroma(persist_directory="./chroma", embedding_function=embedding)

#get llm
os.environ["OPENAI_API_KEY"] = settings.openai_api_key
# llm = GooglePalm(google_api_key=settings.PALM_api_key, temperature=0, max_output_tokens=512)
llm = OpenAI(temperature=1)

#HyDE Prompt_template
hyde_prompt_template = """Please answer the user's question as related to Large Language Models
Question: {question}
Answer:"""

prompt = PromptTemplate(input_variables=["question"], template=hyde_prompt_template)

llm_chain = LLMChain(llm=llm, prompt=prompt)

hyde = HypotheticalDocumentEmbedder(
    llm_chain=llm_chain,
    base_embeddings=embedding
)

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