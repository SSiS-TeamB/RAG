import streamlit as st
import pandas as pd
import numpy as np
from tkinter.tix import COLUMN
# from pyparsing import empty
import time

st.set_page_config(layout='wide')
# add_selectbox = st.sidebar.selectbox("왼쪽 사이드바 Select Box", ("A", "B", "C"))

# 레이아웃
empty1, con1, empty2 = st.columns([0.3, 1.0, 0.3])
empty1, con2, con3, empty2 = st.columns([0.3, 1, 0.5, 0.3])
empty1, con4, empty2 = st.columns([0.3, 1.0, 0.3])
empty1, con5, con6, empty2 = st.columns([0.3, 0.5, 0.5, 0.3])
empty1, con7, empty2 = st.columns([0.3, 1.0, 0.3])


with con1:
   st.title("검색 엔진 시스템")
#    st.header("Header")
#    st.subheader("subheader")
   st.subheader('-'*60)
#    st.header("아래는 (귀여운 아이)")
#    st.image("https://static.streamlit.io/examples/cat.jpg", use_column_width=True)

with con2:
    st.subheader("현재 상황 또는 상태를 적어주시면, 알맞는 복지 서비스를 찾아 드릴게요")
    query_text = st.text_input('아래에 검색하세요')
    # query_text = st.text_area("이건 여러줄 입력")
    if query_text:
        temp = f' "{query_text}"에 대한 정보 검색중...'
        st.write(temp)
        # st.button("click button")
        # st.write("click!!!")
        progress_text = "당신의 기분에 알맞는 서비스 찾는중. 기대하세요."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        my_bar.empty()






##이후에 몇가지 md 를 넣어서 출력 환경 개선하면 된
    # st.button("Rerun")

# with con3:
#     st.header("Chart Data")
#     chart_data = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
#     st.bar_chart(chart_data)
#     st.write("Write Something")


# vector DB Load


