import streamlit as st
import pandas as pd
import numpy as np
import time
from tkinter.tix import COLUMN
from pyparsing import empty


st.set_page_config(layout='wide')
# add_selectbox = st.sidebar.selectbox("왼쪽 사이드바 Select Box", ("A", "B", "C"))

# 레이아웃
empty1, con1, empty2 = st.columns([0.3, 1.0, 0.3])
empty1, con2, con3, empty2 = st.columns([0.3, 0.8, 0.2, 0.3])
empty1, con4, empty2 = st.columns([0.3, 1.0, 0.3])
empty1, con5, con6, empty2 = st.columns([0.3, 0.5, 0.5, 0.3])
empty1, con7, empty2 = st.columns([0.3, 1.0, 0.3])


with con1:
   st.markdown("<h1 style='text-align: center; color: black;'>검색 엔진 시스템</h1>", unsafe_allow_html=True)
#    st.subheader("무엇이든 물어보세요!!!")
   st.markdown("<p style='text-align: right; color: black;'>무엇이든 물어보세요</p>", unsafe_allow_html=True)


#    st.header("Header")
#    st.subheader('-'*60)
#    st.header("A cat")
#    st.image("https://static.streamlit.io/examples/cat.jpg", use_column_width=True)

with con2:
    query_text = st.text_input('검색하셈', label_visibility='collapsed')
    # query_text = st.text_area("이건 여러줄 입력")


with con3:
    # st.header("Chart Data")
    btn_flag = st.button("click")
    # chart_data = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
    # st.bar_chart(chart_data)
    # st.write("Write Something")

with con4:
    if query_text or btn_flag:
        progress_text = f'Finding about "{query_text}"...'
        my_bar = st.progress(0, text=progress_text)

        for i in range(100):
            time.sleep(0.01)
            my_bar.progress(i+1, text=progress_text)
        time.sleep(1)
        my_bar.empty()
        # st.subheader('검색 결과')
        st.markdown("<h2 style='text-align: center; color: black;'>검색 결과</h2>", unsafe_allow_html=True)

# vector DB Load


