"""
    property : sentence_transformers (pip install transformers)

    embedding 쉽게 받아오려고 쓰는거.
    모델 이름은 기본으로 저장된 BAAI/bge-base-en-v1.5 말고 다른거 쓰고싶으면 HuggingFace에서 알아서 찾아서 download에 인수로 넣을 것.

"""

import os
from sentence_transformers import SentenceTransformer


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