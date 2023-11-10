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
        langchain version 생성으로 인한 folder 변경 -> 경로문제 해결해야 함
        (langchain/corpus.py 보고 할 것)

## langchain

    231107 09:19
        corpus 생성하는 함수(llamaindex 결과랑 같이 맞춰줌) 작성 완료 -> 
        {uuid:corpus} 형태로 return함

    231107 12:47
        indent, 의존관계 등 수정했음(langchain만)

    231108 02:24
        OpenAI API update로 Langchain dependency 오류 발생해서 RAG baseline 완성했지만 검증은 못해봄(오류 가능성)

    231108 03:48
        1. document.py 생성으로 이전에 langchain document list 생성과 corpus.py로 uuid 부여한 corpus 생성 통합해서 object로 생성 완료(document.py)
        2. 해당 사항 이용하는 부분들 통합해서 코드 수정 완료

    231108 08:23
        1. qa_generation 완성 -> OpenAI api 업데이트로 langchain과 호환 issue 있어
        추후에 prompt + pydantic 할거면 수정해야 함
        2. main 파일에 dataset 생성 Example Usage 등록

    231108 13:21
        1.document.py에 RegExLoader Class 추가(원문에서 정규식으로 .md file의 table 텍스트 부분)
        2.클래스 쓰는 의미 (self..) 넣었다.
        3.CorpusDBLoader에 with_regex(정규식 사용해서 원문 전처리 할건지) 여부 추가했음.
        
    231109 08:38
        1.document Class BaseDBLoader로 통합하고 flag를 통해서 정규식, split 해결하도록 함 
            -> loader 메서드에 넣음
            (special thanks to khakha👍)
        2.main에 사용예시 변경

    231109 21:10
        1.prompt.py 수정 
        2.임시로 OpenAI 써서 qa_generation.py 완성
        3.내일 평가받고 Dataset 만들기(현재 gpt-3.5-turbo-1106 이용중) -> token limit 16k 낭낭해서 좋음 *^^*

====================================

    231110 14:30
        1. Analogical Prompting 이용해서 prompt.py 수정
        2. 1번의 연장으로 qa_generation.py도 연동되도록 수정
            -> 전체적으로 Prompt Analogical한 방식으로 손질하고 프로세스에 연동되도록 했다.

----> 할 것

1. HyDE Embedding 붙여서 RAG 구현해놓기

2. embedding 불러오는 부분도 object화 해서 구현해놓기

----> 장기적으로..

1. RAG 고도화
2. 어떻게 보여줄 것인가?

