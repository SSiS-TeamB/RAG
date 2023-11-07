"""langchain document 관련해서 db(markdownDB) 불러와서 
    document로 parse하는 과정 object로 생성. """

import os

from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter





class BaseDBLoader:
    def __init__(self, path_db: str = "./markdowndb",):
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

    # def load():


