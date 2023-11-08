# llamaindex

main.py → 실행(1106 구현)

corpus.py → DB 통해서 dataset 생성에 이용하는 corpus 생성

prompt_config.py → prompt 설정

qa_generation → gpt-3.5-turbo(기본값) 이용해서 dataset 생성 후 json에 넣는 것(json 경로 지정)

====================================

231106 09:28
api key 삽입(.gitignore)

qa_generation json 생성 부분 main.py로 옮겨서 실행하도록 함

==> qa_generation 부분 indent 수정할 것