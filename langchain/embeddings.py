"""
    property : sentence_transformers (pip install transformers)

    embedding 쉽게 받아오려고 쓰는거.
    모델 이름은 기본으로 저장된 BAAI/bge-base-en-v1.5 말고 다른거 쓰고싶으면 HuggingFace에서 알아서 찾아서 download에 인수로 넣을 것.

"""
############################################ 임시 : 수정중 1109 ############################################

import os
from sentence_transformers import SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings
import torch

class EmbeddingDownLoader:
    # model 명 기본으로 지정되어 있음. 다운로드 받으려면 변경해야 함.
    def __init__(self, model:str='model_name_here', path=None,) -> None:
        self.model = model #model 이름은 huggingface기준 -> BM-K/KoSimCSE-roberta-multitask
        self.path = path
        return
        
    def download(self):
        # 경로 지정 안해놨으면 현재 스크립트 실행되는 곳에 모델명으로 파일 생성되게 만듦.
        if self.path is None:
            self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.model)
        
        # downloader
        downloader = SentenceTransformer(model_name_or_path=self.model)
        os.makedirs(self.path)
        downloader.save(self.path)

        print(f'model {self.model} downloaded at path {self.path}.')
        return

class EmbeddingLoader:
    def __init__(self) -> None:
        embedding = SentenceTransformerEmbeddings(
        model_name="BM-K/KoSimCSE-roberta-multitask", 
        model_kwargs={'device':self._device_check()}, 
        encode_kwargs={'normalize_embeddings':True},
        )
        return embedding
    def _device_check():
        ''' for check cuda availability '''
        if torch.cuda.is_available() : device ="cuda"
        elif torch.backends.mps.is_available() : device = "mps"
        else : device = "cpu"

        return device

#test
# if __name__ == "__main__":
#     loader = EmbeddingDownLoader(model="BM-K/KoSimCSE-roberta-multitask")
#     loader.download()
