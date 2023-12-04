import streamlit as st
import time
from PIL import Image

# from chromaClient import ChromaClient
from chromaVectorStore import ChromaVectorStore
from rag import RAGPipeline
import math



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

# Settings for semantic_search using vectorstores of langchain
collection_name = "wf_schema_split"
persist_directory = "workspace/chroma_storage"

#### Loading Vectorstore .......
#### 이쪽에 spinner 넣어서 loading check
with st.spinner():
    vectorstore = ChromaVectorStore(**{
    "collection_name":collection_name, 
    "persist_directory":persist_directory,
    "collection_metadata" : {"hnsw:space":"cosine"}
})


with con2:
    query_text = st.text_input('검색하셈', label_visibility='collapsed')
    # query_text = st.text_area("이건 여러줄 입력")

with con3:
    btn_flag = st.button("click")

### button Event
if query_text or btn_flag:
    # semantic_search using "chromadb" module
    # results = chroma_client.semantic_search([query_text], 3)
    start = time.time
 

    
    with con4:
        progress_text = f'Finding about "{query_text}"...'
        with st.status(progress_text, expanded=True) as status:
            st.write("검색 중")
            model = "gpt-3.5-turbo-1106"
            rag_pipeline = RAGPipeline(vectorstore=vectorstore.vs, embedding=vectorstore.emb, model=model)
            if rag_pipeline:
                pass
            results_rag = rag_pipeline.invoke(query_text)
            st.write("관련문서 검색 완료. 답변 생성중")
            time.sleep(2)
            if results_rag:
                pass
            results_vs = rag_pipeline.retrieve(query_text)
            st.write("답변 생성 완료")
            time.sleep(1)
            if results_vs:
                pass
            status.update(label="검색 완료!", state="complete", expanded=False)
            end = time.time
            sec = f"{end-start:.3f} 초"
            st.write(sec)
        # progress bar
        # progress_text = f'Finding about "{query_text}"...'
        # my_bar = st.progress(0, text=progress_text)
        ## RAG result
        # model = "gpt-4-1106-preview"
        # model = "gpt-3.5-turbo-1106"
        # rag_pipeline = RAGPipeline(vectorstore=vectorstore.vs, embedding=vectorstore.emb, model=model)
        # st.toast('답변 생성중!')    
        # for i in range(90):
        #     time.sleep(0.01)
        #     if rag_pipeline:
        #         my_bar.progress(i+1, text=progress_text)

        # st.toast('조금만 더 기다려주세요!')
        # # results_rag = rag_pipeline.invoke(query_text)
        # for i in range(90,95,1):
        #     time.sleep(0.01)
        #     if results_rag:
        #         my_bar.progress(i+1, text=progress_text)
        # # semantic_search using vectorstores of langchain
        # # results_vs = rag_pipeline.retrieve(query_text)
        # for i in range(95,100,1):
        #     time.sleep(0.01)
        #     if results_vs:
        #         my_bar.progress(i+1, text=progress_text)
        
        # st.toast('끝!', icon='🎉')
        # time.sleep(1)
        # my_bar.empty()

        # st.subheader('검색 결과')
        st.markdown("<h2 style='text-align: left; color: white;'>검색 결과</h2>", unsafe_allow_html=True)
    
    
    

    
    with con5:
        st.write("## 답변")
        st.write(results_rag, unsafe_allow_html=True)

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