from deviceCheck import device_check
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings as STE

import os
from sentence_transformers import SentenceTransformer


class EmbeddingLoader:
    def __init__(self, from_default_template: bool = True, **kwargs) -> None:
        """ Embedding model setting. 
            :param from_default_template : 미리 생성해 놓은 모델 이름, SentenceTransformerEmbeddings 설정 불러오는 parameter.        
            """
        #set directory for property Issue
        filedirectory = os.path.dirname(__file__)
        os.chdir(filedirectory)

        if from_default_template:
            kwargs.setdefault("model_name", "da_finetune_epoch_2")
            kwargs.setdefault("model_kwargs", {'device': device_check()})
            kwargs.setdefault("encode_kwargs", {'normalize_embeddings': True})
        self.emb_kwargs = kwargs
        return

    def load(self) -> STE:
        embedding = STE(**self.emb_kwargs)
        return embedding
    
def s_bert_embedding_donwloader(model_name, save_path):
    """
    property : sentence_transformers (pip install transformers)

    embedding Model 쉽게 받아오려고 쓰는거.
    모델 이름은 기본으로 저장된 BAAI/bge-base-en-v1.5 말고 다른거 쓰고싶으면 HuggingFace에서 알아서 찾아서 download에 인수로 넣을 것. 
    """
    s_transformer = SentenceTransformer(model_name_or_path=model_name)

    dir_path = os.path.join(save_path, model_name)
    print(f'make {dir_path} directory!')
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    print(f'{model_name} download')
    s_transformer.save(dir_path)
    
    print(f'model {model_name} downloaded at path {dir_path}.')
    return 

print(EmbeddingLoader().load())