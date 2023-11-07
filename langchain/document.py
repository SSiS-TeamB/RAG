"""langchain document 관련해서 db(markdownDB) 불러와서 
    document로 parse하는 과정 object로 생성."""

import os
import uuid

from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter

class BaseDBLoader:
    """markdownDB folder에서 불러온 다음에 폴더별로 내부에 있는 내용 Load해서 Split하고 저장함"""
    def __init__(self, path_db:str="./markdowndb"):
        #textsplitter config
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=200,
            is_separator_regex=False,
        )
        
        #directoryconfig
        self.directory = os.path.dirname(__file__)
        os.chdir(self.directory)

        #get list of db
        self.path_db = path_db
        self.list_of_path_db = os.listdir(path_db)

    def load(self)->list:
        """Generate corpus from langchain document object"""
        list_of_path_db=self.list_of_path_db
        text_splitter=self.text_splitter
        
        result_storage = []

        for db_folder in list_of_path_db:
            directory_loader = DirectoryLoader(path=os.path.join(self.path_db, db_folder), loader_cls=UnstructuredMarkdownLoader)
            result = directory_loader.load_and_split(text_splitter=text_splitter)
            result_storage.extend(result)     

        return result_storage

class CorpusDBLoader(BaseDBLoader):
    def _result_to_corpus(self, corpus:dict, result:list)->dict:
        """Get langchain Document object and convert to dict form with uuid."""
        for document in result:
            corpus[str(uuid.uuid4())] = document.page_content
        return corpus
    
    def load(self, corpus:dict={})->dict:
        """Generate corpus from langchain document object"""
        list_of_path_db = self.list_of_path_db
        text_splitter = self.text_splitter

        for db_folder in list_of_path_db:
            directory_loader =  DirectoryLoader(path=os.path.join("./markdowndb", db_folder), loader_cls=UnstructuredMarkdownLoader)
            result = directory_loader.load_and_split(text_splitter=text_splitter)
            corpus_piece = self._result_to_corpus({}, result)
            corpus.update(corpus_piece)
        return corpus