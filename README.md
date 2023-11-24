# 의견 Note
- 코드 이해 + Web 추가 기능 넣기

## 2023-11-23
- 수정 내용 : metadata 확인해서 관련문서 가져온 파일경로(source)넣고 관련 url도 추가
    - rag.py
        ```
        @staticmethod
        def format_docs(docs:list[Document]):
        ```
    - 추가 파일
        - workspace/meta_excel.xlsx     : 위치 꼭 여기로 해줘야됨,,
            - columns = ['source','url']
                - source : 관련문서 가져온 파일경로(앞 경로는 짜르고 뒤에 필요한 내용만 있음)
                - url    : 관련문서에 대한 url