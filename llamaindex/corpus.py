import os

from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
#MetaDataMode
from llama_index.schema import MetadataMode


def create_corpus(dbpath:str="./markdownDB") -> dict:
    """ DB 경로 받아서 corpus 생성하는 함수 (Dataset 생성용). """
    # list db and get text from each file in it -> to llama_index.schema.Document Object
    list_markdown_db = os.listdir(dbpath)

    # only .md file
    required_exts=[".md"]
    docs = []

    # iterate each folder -> Read and Load each file -> store it to docs.
    for directory_db in list_markdown_db:
        if directory_db=="readme.md": 
            continue

        absolute_path_db = os.path.abspath(os.path.join(f'{dbpath}/', directory_db))

        # set reader to each folder
        reader = SimpleDirectoryReader(
            input_dir=absolute_path_db,
            required_exts=required_exts,
            recursive=True,
            encoding="utf-8",
        )
        docs.extend(reader.load_data())

    print(f"Loaded {len(docs)} docs")

    # NodeParser -> LangChain에서는 text_splitter
    node_parser=SimpleNodeParser.from_defaults()
    nodes = node_parser.get_nodes_from_documents(documents=docs, show_progress=True)
    print(f"Create {len(nodes)} nods from documents")

    # Metadata 무시한 상태로 Node 기준으로 부여된 id로 corpus 생성.
    corpus = {node.node_id : node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes}
    
    return corpus


#test
if __name__ == "__main__" :
    print(create_corpus())