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
    """ MultiThreading API calling Executor """
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
    #Query example for user
    st.subheader("📌이렇게 검색해보세요!")
    st.info('''
            예시)\n
            ❓"대학생인데 학자금 대출 말고도 금전적인 지원을 받을 수 있는 방법이 있나요?"\n
            ❓"노인 분들이 문화생활을 즐길 수 있는 공간으로, 어떤 시설이 있는지 궁금해요."\n
            ❓"우리 집이 최근 화재로 인해 살 곳을 잃었어요. 긴급하게 지원받을 수 있는 방법이 있을까요?"
            ''')

    return

def vectorstore_config():
    """ VectorStore Config """
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
    ##### LLM container
    llm_selector = st.container()
    with llm_selector:
        st.write("")
        st.subheader("⚙️답변 생성 AI 모드 설정")
        option = st.selectbox(
            "정확한 답변 생성은 조금 느릴 수 있어요.",
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
    

    # 필터링
    category_list = ["01 생계 지원", "02 취업 지원", "03 임신·보육 지원", "04 청소년·청년 지원", "05 보건의료 지원", "06 노령층 지원", "07 장애인 지원", "08 보훈대상자 지원", "09 법률·금융 복지 지원", "10 기타 위기별·상황별 지원"]
    filter_len = len(category_list)
    chk_idxes = [True]*filter_len
    meta_filter = st.container()
    with meta_filter:
        st.subheader("⚙️필터 적용")
        with st.expander("주제 선택"):
            for i in range(filter_len):
                chk_idxes[i] = st.checkbox(f'{category_list[i]}', value=True)
        st.write("")
    # filter_dict={'category': {"$eq": '02 취업 지원'}}
    # filter_dict = {"$or": [{"title": {"$eq": "내일이룸학교"}}, {"title": {"$eq": "일학습병행제"}}]}
    filter_dict = {}
    chk_num = sum(chk_idxes)
    if chk_num == 0:
        st.warning("카테고리를 1개 이상 선택해주세요!", icon="⚠️")
        return
    elif chk_num == 1:
        for i in range(filter_len):
            if chk_idxes[i]:
                filter_dict["category"] = {"$eq": category_list[i]}
                break
    else:
        filter_dict["$or"] = []
        for i in range(filter_len):
            if chk_idxes[i]:
                filter_dict["$or"].append({"category": {"$eq": category_list[i]}})

    ##### METHOD RESULT CONTAINER
    # invoke container
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

    ##### stream test
    pipeline = RAGPipeline(vectorstore=vectorstore.vs, embedding=vectorstore.emb, model=model, filter_dict=filter_dict)
    
    #ON Button Event
    if query or search_button:
        invoke_empty.markdown("실행 중 ... ")
        with ThreadPoolExecutor() as executor:
            future_invoke = executor.submit(run_pipeline_task, query, pipeline.invoke)
            future_retrieve = executor.submit(run_pipeline_task, query, pipeline.retrieve)
            futures = {future_invoke: 'Invoke', future_retrieve: 'Retrieve'}
            # futures = {future_retrieve: 'Retrieve'}

            for future in as_completed(futures):
                task_name = futures[future]
                result, elapsed_time = future.result()

                if task_name == 'Invoke':
                    invoke_empty.empty()
                    invoke_empty.write({"답변 내용": f'{result} \n\n실행 시간: {elapsed_time:.2f}초'})
                    # invoke_empty.markdown(f'{result} \n\n 실행 시간: {elapsed_time:.2f}초', unsafe_allow_html=True)
                    pass
                else:
                    empty = retrieve_container.empty()
                    relavant_docs_list = RAGPipeline.format_docs(result).split("\n***\n")
                    # empty.markdown(relavant_docs_list, unsafe_allow_html=True)
                    with empty.container():
                        for f_doc in relavant_docs_list:
                            h, c, t = f_doc.split("$$$")
                            st.markdown(h, unsafe_allow_html=True)
                            st.write({"상세보기": c})
                            st.markdown(t, unsafe_allow_html=True)
                            st.divider()
                        st.write(f'실행 시간: {elapsed_time:.2f}초')

            # for t in executor._threads:
            #     add_script_run_ctx(t)
        st.toast('검색 완료!', icon="✅")

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