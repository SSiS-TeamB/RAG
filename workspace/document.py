"""langchain document 관련해서 db(markdownDB) 불러와서 
    document로 parse하는 과정 object로 생성."""

import os
import uuid
import re

from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

class BaseDBLoader:
    """markdownDB folder에서 불러온 다음에 폴더별로 내부에 있는 내용 Load해서 Split하고 저장함"""

    def __init__(self, loader_cls=UnstructuredMarkdownLoader, path_db: str = "./markdowndb"):
        # textsplitter config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            is_separator_regex=False,
        )
        
        #필요한 이유 -> llamaindex와 같이 폴더가 나누어져 있는 경우 경로 잡을 때 오류 발생할 가능성 줄이려고.
        self.directory = os.path.dirname(__file__)
        os.chdir(self.directory)

        # loaderclass config
        self.loader_cls = loader_cls
        # md 파일 담고 있는 전체 directory 경로
        self.path_db = path_db
        # storage
        self.storage = []

        return

    def load(self, is_split=True, is_regex=False) -> list[Document]:
        """Generate corpus from langchain document objects"""
        for db_folder in os.listdir(self.path_db):
            db_folder_abs = os.path.join(self.path_db, db_folder)
            directory_loader = DirectoryLoader(path=db_folder_abs, loader_cls=self.loader_cls)

            if is_split:
                result = directory_loader.load_and_split(text_splitter=self.text_splitter)
            else:
                result = directory_loader.load()

            self.storage.extend(result)
        #정규식 여부에 따라서 storage 변경
        if is_regex:
            self.storage = self._result_to_regex()

        return self.storage
    
    def _result_to_regex(self) -> list:
        regex = '([^가-힣0-9a-zA-Z.,·•%↓()\s\\\])'

        result = []
        for document in self.storage:
            sub_str = re.sub(pattern=regex, repl="", string=document.page_content)
            document.page_content = sub_str
            result.append(document)

        self.storage = result

        return self.storage

    def get_corpus(self) -> dict:
        "self.storage가 존재한다면 dict로 결과 return함"
        if not self.storage :
            raise ValueError("loader에 storage가 생성되지 않았습니다. load 함수를 실행하거나 storage를 확인하고 다시 실행하세요.")
        
        return {str(uuid.uuid4()): doc.page_content for doc in self.storage}