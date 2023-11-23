import streamlit as st
import time
from PIL import Image

# from chromaClient import ChromaClient
from chromaVectorStore import ChromaVectorStore
from rag import RAGPipeline


st.set_page_config(layout='wide')
# add_selectbox = st.sidebar.selectbox("왼쪽 사이드바 Select Box", ("A", "B", "C"))

# 레이아웃
backgroundColor = "#F0F0F0"
empty1, con1, empty2 = st.columns([0.3, 1.0, 0.3])
empty1, con2, con3, empty2 = st.columns([0.3, 0.8, 0.2, 0.3])
empty1, con4, empty2 = st.columns([0.3, 1.0, 0.3])
empty1, con5, con6, empty2 = st.columns([0.3, 0.5, 0.5, 0.3])
empty1, con7, empty2 = st.columns([0.3, 1.0, 0.3])

# Settings for semantic_search using vectorstores of langchain
# vs_info_dict = {"collection_name":"wf_schema", "persist_directory":"workspace/chroma_storage",}
# vs_info_dict = {"collection_name":"wf_schema_no_split", "persist_directory":"workspace/chroma_storage",}
# vector_store = ChromaVectorStore(**vs_info_dict)

with con1:
    st.markdown("<h1 style='text-align: center; color: gray;'>검색 엔진 시스템</h1>", unsafe_allow_html=True)
    img_ssis = Image.open('image/ssis_logo.png')
    img_BL = Image.open('image/bigleader_logo.png')
    empty1, col3, col2, col1 = st.columns([3, 0.8, 0.1, 1.2])
    col1.image(img_ssis, use_column_width=True)
    col2.empty()
    col3.image(img_BL, use_column_width=True)
    st.markdown("<p style='text-align: right; color: gray;'>무엇이든 물어보세요</p>", unsafe_allow_html=True)
#    st.header("Header")
#    st.image("https://static.streamlit.io/examples/cat.jpg", use_column_width=True)

with con2:
    query_text = st.text_input('검색하셈', label_visibility='collapsed')
    # query_text = st.text_area("이건 여러줄 입력")

with con3:
    btn_flag = st.button("click")

# Settings for semantic_search using vectorstores of langchain
collection_name = "wf_schema_split"
persist_directory = "workspace/chroma_storage"

#### Loading Vectorstore .......

with st.spinner():
    vectorstore = ChromaVectorStore(**{
    "collection_name":collection_name, 
    "persist_directory":persist_directory,
    "collection_metadata" : {"hnsw:space":"cosine"}
})

#### 이쪽에 spinner 넣어서 loading check


### button Event
if query_text or btn_flag:
    # semantic_search using "chromadb" module
    # results = chroma_client.semantic_search([query_text], 3)

    ## RAG result
    model = "gpt-4-1106-preview"
    # model = "gpt-3.5-turbo-1106"
    rag_pipeline = RAGPipeline(vectorstore=vectorstore.vs, embedding=vectorstore.emb, model=model)
    results_rag = rag_pipeline.invoke(query_text)

    # semantic_search using vectorstores of langchain
    results_vs = rag_pipeline.retrieve(query_text)

    with con4:
        # progress bar
        progress_text = f'Finding about "{query_text}"...'
        my_bar = st.progress(0, text=progress_text)

        for i in range(100):
            time.sleep(0.01)
            my_bar.progress(i+1, text=progress_text)
        time.sleep(1)
        my_bar.empty()

        # st.subheader('검색 결과')
        st.markdown("<h2 style='text-align: left; color: white;'>검색 결과</h2>", unsafe_allow_html=True)

    with con5:
        st.write("## 답변")
        st.markdown(results_rag)

    with con6:
        st.markdown("## 관련 문서")
        st.markdown(RAGPipeline.format_docs(results_vs))
        # sp_str = "\n"+'*'*50+"\n"
        # st.write(sp_str.join(doc.page_content for doc in results_vs))
        # print(results_vs)

   
        # print("There are", vectorstore._collection.count(), "in the collection.")
        # results = run_search(search_query)

        # for result in results :
        #     st.title(f'**{result.metadata["title"]}**')
        #     st.markdown(result.page_content)
        #     st.write(result.metadata['tag'].split(','))