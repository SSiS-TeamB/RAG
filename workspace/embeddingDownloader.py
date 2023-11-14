"""
    property : sentence_transformers (pip install transformers)

    embedding 쉽게 받아오려고 쓰는거.
    모델 이름은 기본으로 저장된 BAAI/bge-base-en-v1.5 말고 다른거 쓰고싶으면 HuggingFace에서 알아서 찾아서 download에 인수로 넣을 것.

"""

import os
from sentence_transformers import SentenceTransformer

# class EmbeddingDownLoader:
#     # model 명 기본으로 지정되어 있음. 다운로드 받으려면 변경해야 함.
#     def __init__(self, model:str='model_name_here', path=None,) -> None:
#         self.model = model #model 이름은 huggingface 기준 -> BM-K/KoSimCSE-roberta-multitask
#         self.path = path
#         return
        
#     def download(self):
#         # 경로 지정 안해놨으면 현재 스크립트 실행되는 곳에 모델명으로 파일 생성되게 만듦.
#         if self.path is None:
#             self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.model)
        
#         # downloader
#         downloader = SentenceTransformer(model_name_or_path=self.model)
#         os.makedirs(self.path)
#         downloader.save(self.path)

#         print(f'model {self.model} downloaded at path {self.path}.')
#         return

#test
# if __name__ == "__main__":
#     loader = EmbeddingDownLoader(model="BM-K/KoSimCSE-roberta-multitask")
#     loader.download()

def s_bert_embedding_downloader(model_name, save_path):
    s_transformer = SentenceTransformer(model_name_or_path=model_name)

    dir_path = os.path.join(save_path, model_name)
    print(f'make {dir_path} directory!')
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    print(f'{model_name} download')
    s_transformer.save(dir_path)
    
    print(f'model {model_name} downloaded at path {dir_path}.')
    return 