import streamlit as st
import time
from PIL import Image

# from chromaClient import ChromaClient
from chromaVectorStore import ChromaVectorStore
from rag import RAGPipeline

# def launch():
    # my_bar = st.progress(0, text="검색 중입니다. 잠시만 기다려주세요.") -> 검색 버튼 누르고 바로 실행
    #my_bar.progress(percent_complete + 1, text=progress_text) -> 어디다 넣을지 고민
    #my_bar.empty()

st.set_page_config(
page_title="Welfare Search Serviece",
page_icon="✨",
layout="centered",
initial_sidebar_state="collapsed",
)

backgroundColor = "#F0F0F0"
title = '''<h1 style='text-align: center'>복지 정보 검색 서비스</h1><br>
<center>나에게 딱 맞는 복지 정보<br>
이제는 누구나 쉽게, 내 마음대로 검색할 수 있어요!</center><br>
'''
st.markdown(title, unsafe_allow_html=True)
st.divider()

#Query example for user
st.subheader("📌이렇게 검색해보세요!")
st.info('예시: "20대 취업관련 제도"')

#Select gpt version
with st.container():
    st.subheader("⚙️검색 모드 선택")
    option_speed, option_accuracy = st.columns([0.2, 0.8])
    gpt_3_5 = option_speed.button("빠른 검색")
    gpt_4 = option_accuracy.button("정확한 검색")

    if gpt_3_5:
        model = "gpt-3.5-turbo-1106"
        search_name = "빠른 검색"
        option = True
    elif gpt_4:
        model = "gpt-4-1106-preview"
        search_name = "정확한 검색"
        option = True
    else:
        st.error('검색 모드를 선택하세요.', icon="🚨")
        search_name = "검색 모드를 선택 하세요."
        option = False
    st.divider()

#Enter the query
query_text = st.text_input("Search Bar", placeholder="검색어를 입력하세요.", label_visibility="hidden")
search_buttion = st.button(search_name, use_container_width=True, disabled=(option is False))

#button Event
if query_text or search_buttion:
    # Settings for semantic_search using vectorstores of langchain
    collection_name = "wf_schema_split"
    persist_directory = "workspace/chroma_storage"            

    vectorstore = ChromaVectorStore(**{
    "collection_name":collection_name, 
    "persist_directory":persist_directory,
    "collection_metadata" : {"hnsw:space":"cosine"}
    })

    # semantic_search using "chromadb" module
    # results = chroma_client.semantic_search([query_text], 3)
    
    ## RAG result
    rag_pipeline = RAGPipeline(vectorstore=vectorstore.vs, embedding=vectorstore.emb, model=model)
    results_rag = rag_pipeline.invoke(query_text)

    # semantic_search using vectorstores of langchain
    results_vs = rag_pipeline.retrieve(query_text)
    
    #Get Answer
    with st.container():
        st.divider()
        st.subheader(f"{search_name} 결과")
        st.markdown(results_rag)
        st.divider()
        st.markdown("## 관련 문서")
        st.markdown(RAGPipeline.format_docs(results_vs))

# # launch
# if __name__  == "__main__" :
#     # device_check()
#     launch()