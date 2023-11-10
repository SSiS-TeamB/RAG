import streamlit as st
import pandas as pd
import numpy as np
from tkinter.tix import COLUMN
from pyparsing import empty


st.set_page_config(layout='wide')
# add_selectbox = st.sidebar.selectbox("왼쪽 사이드바 Select Box", ("A", "B", "C"))

# 레이아웃
empty1, con1, empty2 = st.columns([0.3, 1.0, 0.3])
empty1, con2, con3, empty2 = st.columns([0.3, 0.5, 0.5, 0.3])
empty1, con4, empty2 = st.columns([0.3, 1.0, 0.3])
empty1, con5, con6, empty2 = st.columns([0.3, 0.5, 0.5, 0.3])
empty1, con7, empty2 = st.columns([0.3, 1.0, 0.3])


with con1:
   st.title("Title")
   st.header("Header")
   st.subheader("subheader")
   st.subheader('-'*60)
#    st.header("A cat")
#    st.image("https://static.streamlit.io/examples/cat.jpg", use_column_width=True)

with con2:
    st.header("Button")
    query_text = st.text_input('검색하셈')
    # query_text = st.text_area("이건 여러줄 입력")
    if query_text:
        temp = f'Finding about "{query_text}"...'
        st.write(temp)
    if st.button("click button"):
        st.write("click!!!")

with con3:
    st.header("Chart Data")
    chart_data = pd.DataFrame(np.random.randn(50, 3), columns=["a", "b", "c"])
    # st.bar_chart(chart_data)
    st.write("Write Something")


# vector DB Load


