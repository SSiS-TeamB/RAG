""" 1116 13:18 dependency 해결. deviceCheck 삭제하고 EmbeddingLoader 안에 통합해서 해결. """

import os
import torch

from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings as STE

from sentence_transformers import SentenceTransformer


class EmbeddingLoader:
    def __init__(self, **kwargs) -> None:
        """ Embedding model setting. 
            
            args
                model_name: model name or path(HuggingFaceEmbeddings)
                encode_kwargs: encoding kwargs from SentenceTransformer model kwargs
        """
        kwargs.setdefault("model_kwargs", {'device': self._device_check()})
        self.kwargs = kwargs
        return

    def load(self) -> STE:
        embedding = STE(**self.kwargs)
        return embedding
    
    def _device_check(self) -> str: 
        ''' for check cuda availability '''
        if torch.cuda.is_available(): device = "cuda"
        elif torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
        return device
    
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


## test (안 되는 이유 -> deviceCheck 삭제하고 EmbeddingLoader 안에 통합해서 해결)

if __name__ == "__main__" :
    # setup = EmbeddingLoader(from_default_template=False, model_name="workspace/dadt_epoch2_kha_tok", encode_kwargs = {'normalize_embeddings': True}).load()
    setup = EmbeddingLoader(model_name="workspace/dadt_epoch2_kha_tok", encode_kwargs = {'normalize_embeddings': True})