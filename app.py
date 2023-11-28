import time
import streamlit as st
from chromaVectorStore import ChromaVectorStore
from rag import RAGPipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit.runtime.scriptrunner import add_script_run_ctx
from PIL import Image

### MultiThread로 실행하기 위해 실행요소만 따로 뺐다.
## RAGPipeline.invoke, RAGPipeline.retrieve -> 실행 후 time check
def run_pipeline_task(query, task_func):
    start_time = time.time()
    try:
        result = task_func(query)
    except Exception as e:
        result = str(e)
    elapsed_time = time.time() - start_time
    # print(elapsed_time)
    return result, elapsed_time

def page_config():
    """ Execute page config """
    st.set_page_config(
    page_title="Welfare Search Serviece",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed",
    )
    #Title
    title = '''<h1 style='text-align: center'>복지 정보 검색 서비스</h1><br>
    <center>나에게 딱 맞는 복지 정보<br>
    이제는 누구나 쉽게, 내 마음대로 검색할 수 있어요!</center><br>
    '''
    st.markdown(title, unsafe_allow_html=True)
    #Image
    img1, img2 = st.columns(2)
    with img1:
        img_ssis = Image.open('image/ssis_logo.png')
        img1.image(img_ssis, use_column_width=True)
    with img2:
        img_BL = Image.open('image/bigleader_logo.png')
        img2.image(img_BL, use_column_width=True)
    st.subheader("", divider='blue')

def vectorstore_config():
    # Settings for semantic_search using vectorstores of langchain
    collection_name = "wf_schema_split"
    persist_directory = "workspace/chroma_storage"            

    vectorstore = ChromaVectorStore(**{
    "collection_name":collection_name, 
    "persist_directory":persist_directory,
    "collection_metadata" : {"hnsw:space":"cosine"}
    })

    return vectorstore

def main() :
    #load page config
    page_config()

    #Query example for user
    st.subheader("📌이렇게 검색해보세요!")
    st.info('예시: "20대 취업관련 제도"')

    ##### CONTAINERS
    #LLM container
    llm_selector = st.container()
    with llm_selector:
        st.write("")
        st.subheader("⚙️답변 생성 AI 모드 설정")
        option = st.selectbox(
            "더 정확한 답변 생성은 조금 느릴 수 있어요.",
            ('빠른 생성', '정확한 생성'),
            label_visibility="visible",
        )
        # option_speed, option_accuracy = st.columns([0.2, 0.8])
        # gpt_3_5 = option_speed.button("빠른 검색")
        # gpt_4 = option_accuracy.button("정확한 검색")
        if option == '빠른 생성':
            model = "gpt-3.5-turbo-1106"
            search_name = "빠른 생성"
        else:
            model = "gpt-4-1106-preview"
            search_name = "정확한 생성"
        
    ##### QUERY CONFIG
    query_container = st.container()
    with query_container:
        st.subheader("🔍검색")
        query = st.text_input("Search Bar", placeholder="검색어를 입력하세요.", label_visibility="hidden")
        search_button = st.button(search_name, use_container_width=True)
        st.divider()
    
    ##### METHOD RESULT CONTAINER
    #invoke container
    invoke_container = st.container()
    with invoke_container:
        st.markdown("### 생성된 답변")
        invoke_empty = st.empty()
        st.divider()
    #retrieve container
    retrieve_container = st.container()
    with retrieve_container:
        st.markdown("### 관련 문서")  
    ##### VECTORSTORE CONFIG
    vectorstore = vectorstore_config()

    #ON Button Event
    if query or search_button:
        model = "gpt-4-1106-preview"
        pipeline = RAGPipeline(vectorstore=vectorstore.vs, embedding=vectorstore.emb, model=model)

        invoke_empty.markdown("실행 중 ... ")
        with ThreadPoolExecutor() as executor:
            future_invoke = executor.submit(run_pipeline_task, query, pipeline.invoke)
            future_retrieve = executor.submit(run_pipeline_task, query, pipeline.retrieve)
            futures = {future_invoke: 'Invoke', future_retrieve: 'Retrieve'}

            for future in as_completed(futures):
                task_name = futures[future]
                result, elapsed_time = future.result()

                if task_name == 'Invoke':
                    invoke_empty.empty()
                    invoke_empty.markdown(f'{result} \n\n 실행 시간: {elapsed_time:.2f}초', unsafe_allow_html=True)

                else:
                    empty = retrieve_container.empty()
                    empty.markdown(f'{RAGPipeline.format_docs(result)} \n\n 실행 시간: {elapsed_time:.2f}초')
            for t in executor._threads:
                add_script_run_ctx(t)
        st.success("검색 완료!")

        # progress_text = f'Finding about "{query_text}"...'
        # with st.spinner(progress_text):
        #     placeholder = st.empty()
        #     a = "검색중"
        #     placeholder.text(a)
            
        #     ## RAG result
        #     pipeline = RAGPipeline(vectorstore=vectorstore.vs, embedding=vectorstore.emb, model=model)
        #     results_rag = pipeline.invoke(query_text)
        #     a = "관련문서 검색 완료. 답변 생성중"
        #     placeholder.text(a)
        #     time.sleep(2)
        #     # semantic_search using vectorstores of langchain
        #     results_vs = pipeline.retrieve(query_text)
        #     a = "답변 생성 완료"
        #     placeholder.text(a)
        #     time.sleep(1)
        #     placeholder.empty()
        #     st.success("검색 완료!")
            
        # #Get Answer
        # answer, docs = st.tabs([f"{search_name} 결과", "관련 제도"])
        # with answer:
        #     st.subheader(f'''
        #                 "{query_text}"에 대한 **:blue[{search_name}]** 결과입니다.''')
        #     st.write("")
        #     st.markdown(results_rag)
        # with docs:
        #     st.subheader(f'"{query_text}" 관련 복지 제도입니다.')
        #     st.markdown(RAGPipeline.format_docs(results_vs))


#### run -> streamlit run app.py
if __name__ == "__main__":
    main()