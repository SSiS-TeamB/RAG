"""langchain document 관련해서 db(markdownDB) 불러와서 
    document로 parse하는 과정 object로 생성."""

import os
import uuid
import re

from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document


def result_to_regex(result: list[Document]) -> list:
    regex = '([^가-힣0-9a-zA-Z.,·•%↓()\s\\\])'

    storage = []
    for document in result:
        sub_str = re.sub(pattern=regex, repl="$$$$", string=document.page_content)
        document.page_content = sub_str
        storage.append(document)
    return storage


class BaseDBLoader:
    """markdownDB folder에서 불러온 다음에 폴더별로 내부에 있는 내용 Load해서 Split하고 저장함"""

    def __init__(self, loader_cls=UnstructuredMarkdownLoader, path_db: str = "./markdowndb", ):
        # textsplitter config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            is_separator_regex=False,
        )
        # loaderclass config
        self.loader_cls = loader_cls
        # md 파일 담고 있는 전체 디렉터리 경로
        self.path_db = path_db
        return

    def load(self, is_split=True, is_regex=False) -> list[Document]:
        """Generate corpus from langchain document objects"""

        result_storage = []
        for db_folder in os.listdir(self.path_db):
            db_folder_abs = os.path.join(self.path_db, db_folder)
            directory_loader = DirectoryLoader(path=db_folder_abs, loader_cls=self.loader_cls)
            if is_split:
                result = directory_loader.load_and_split(text_splitter=self.text_splitter)
            else:
                result = directory_loader.load()

            result_storage.extend(result)

        if is_regex:
            result_storage = result_to_regex(result_storage)

        return result_storage

    def get_corpus(self) -> dict:
        return {str(uuid.uuid4()): doc.page_content for doc in self.load(is_split=True, is_regex=False)}
