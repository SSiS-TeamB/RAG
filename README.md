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

23-11-07 09:19

corpus 생성하는 함수(llamaindex 결과랑 같이 맞춰줌) 작성 완료 -> {uuid:corpus} 형태로 return함