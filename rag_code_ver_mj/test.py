"""langchain document 관련해서 db(markdownDB) 불러와서 
    document로 parse하는 과정 object로 생성."""

import os
import uuid

from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter

class BaseDBLoader:
    """markdownDB folder에서 불러온 다음에 폴더별로 내부에 있는 내용 Load해서 Split하고 저장함"""
    
    # text load & split  에 필요한 configuration 
    # text load 전 디렉토리 설정 후 text file list 생성
    def __init__(self, path_db:str="./MarkDownDB"):
        #textsplitter config
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=200,
            is_separator_regex=False,  
            # seperator -> default : ["\n\n", "\n", " ", ""]  , seperator 지정시  졍규표현식인지 여부 확인
        )
        
        #directoryconfig
        ##__file__이 실행되는 디렉토리 반환
        self.directory = os.path.dirname(__file__)
        os.chdir(self.directory)

        #get list of db
        # path_db 에 있는 파일들 리스트로 저장
        self.path_db = path_db
        self.list_of_path_db = os.listdir(path_db)

    ##  text load & split
    def load(self)->list:
        """Generate corpus from langchain document object"""
        list_of_path_db=self.list_of_path_db
        text_splitter=self.text_splitter
        
        ## markdownDB 에 있는 파일들 load_split 후 저장할 리스트 생성
        result_storage = []

        ## path_db 파일 리스트에 있는 파일 하나씩 불러와서 load 후 split 해서 extend
        for db_folder in list_of_path_db:
            directory_loader = DirectoryLoader(path=os.path.join(self.path_db, db_folder), loader_cls=UnstructuredMarkdownLoader)
            result = directory_loader.load_and_split(text_splitter=text_splitter)
            result_storage.extend(result)     

        return result_storage


## split 되어 저장된 파일을 기반으로 corpus 생성
# BaseDBLoader 상속받은 자식클래스
# uuid 부여해서 corpus 생성
class CorpusDBLoader(BaseDBLoader):
    def _result_to_corpus(self, corpus:dict, result:list)->dict:
        """Get langchain Document object and convert to dict form with uuid."""
        
        ## split 후 각 문서에 uuid부여하여 만든 corpus 리턴
        for document in result:
            corpus[str(uuid.uuid4())] = document.page_content  ## documnet 의 text 가져와서 uuid 부여
        return corpus
    
    ## BaseDBLoader 의 load method를 overriding
    def load(self, corpus:dict={})->dict:
        """Generate corpus from langchain document object"""
        
        # split 전 md파일들 리스트에 저장
        # splitter config(환경설정)
        list_of_path_db = self.list_of_path_db
        text_splitter = self.text_splitter

        # split안된 md파일들 불러와서 split 후 document별로 uuid부여 -> corpus에 update
        # 10대 대분류 -> 각 욕구별로 불러옮
        for db_folder in list_of_path_db:
            directory_loader =  DirectoryLoader(path=os.path.join("./markdownDB", db_folder), loader_cls=UnstructuredMarkdownLoader)
            ##loader_cls : 파일 불러오기 위한 클래스 지정
            #chunk_size에 맞게 split되서 result에 document로 저장
            #type : langchain.schema.document.Document
            result = directory_loader.load_and_split(text_splitter=text_splitter)  #load 후 split 바로 진행 
            corpus_piece = self._result_to_corpus({}, result) # 임시 corpus ={}  ==> 빈 딕셔너리로 초기화
            corpus.update(corpus_piece) # uuid 부여된 corpus_piece 를 corpus에 update
        return corpus
    

test_Corpus = CorpusDBLoader()
corpus = test_Corpus.load()
print(corpus)
print("="*100)
print(type(corpus))