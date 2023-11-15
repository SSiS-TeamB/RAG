# 의견 Note

## 231114
#### 송준호 : 가재준씨 통합으로 만들어서 main에 올렸습니다   
    1. workspace/main.py 삭제(QAgen에 사용하는거라 삭제했음)   
    2. workspace/prompt.py 삭제(QAgen에 사용하는거라 삭제했음)
    3. workspace/rag.py 수정(231113 최신버전 반영하였음 -> joonho branch)
    4. streamlit_app.py 이름 mainView.py로 변경(합의된 사항임)   
    5. chromaClient.py line 30~33에 의견 주석 달았음

    -> 일단 이 구조로 가는거고 추후에 통합사항 더 논의해서 바꾸면 될 듯

#### 
    1. workspace/analogicalPrompt.py 작성 -> RAG generator Prompt

## 231115
#### 송준호
    1. workspace/embeddingSetup.py 만들어서 EmbeddingLoading 객체화
    2. workspace/prompt 생성, analogicalPrompting.py 연동으로 codeline 줄임
    3. rag 연동되도록 수정
    4. chromaClient.py, chromaVectorStore.py 중복된 기능 chromaVectorStore.py 한 클래스(ChromaVectorStore) 로 통합

#### 가재준
    1. chromaClient.py ChromaClient class 내부 메소드들 구현
    2. chromaVectorStore.py Document 받고 -> 답변 retrieve 하는 class Cleancode로 수정
    3. mainView.py Cleancode