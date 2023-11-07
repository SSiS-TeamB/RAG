# Update Note

## llamaindex

main.py → 실행(1106 구현)

corpus.py → DB 통해서 dataset 생성에 이용하는 corpus 생성

prompt_config.py → prompt 설정

qa_generation → gpt-3.5-turbo(기본값) 이용해서 dataset 생성 후 json에 넣는 것(json 경로 지정)

====================================

231106 09:28

api key 삽입(.gitignore)

qa_generation json 생성 부분 main.py로 옮겨서 실행하도록 함

====================================

231107 09:19

langchain version 생성으로 인한 folder 변경 -> 경로문제 해결해야 함(langchain/corpus.py 보고 할 것)

## langchain

231107 09:19

corpus 생성하는 함수(llamaindex 결과랑 같이 맞춰줌) 작성 완료 -> {uuid:corpus} 형태로 return함

231107 12:47

indent, 의존관계 등 수정했음(langchain만)

====================================

231108 02:24

----> 할 것
    
    1.HyDE Embedding 붙여서 RAG 구현해놓기

    2.langchain generator 완성하기

    3.쓸데없는 부분 object ㄱㄱ

OpenAI API update로 Langchain dependency 오류 발생해서 RAG baseline 완성했지만 검증은 못해봄(오류 가능성)

