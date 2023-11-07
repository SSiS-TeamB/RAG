import os

from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#TextSplitter config
text_splitter=RecursiveCharacterTextSplitter(
    separators="\n\n",
    chunk_size=200,
    is_separator_regex=False,
)

#directoryconfig
directory = os.path.dirname(__file__)
os.chdir(directory)

#get list of db
list_of_path_db = os.listdir("./markdowndb")

for db_folder in list_of_path_db:
    directory_loader = DirectoryLoader(path=os.path.join("./markdowndb", db_folder), loader_cls=UnstructuredMarkdownLoader)
    result=directory_loader.load_and_split(text_splitter=text_splitter)

#chroma setup


