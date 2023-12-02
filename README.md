![logo](/image/bluebird.png)

# SSiS 검색엔진 개선ver (TEAM BlueBird)

### I. 연구 배경 및 목적

- 기존의 [복지로](https://www.bokjiro.go.kr/) 검색 서비스의 최대 난점은 키워드 기반 검색(Keyword Based Search)에 따른 문맥 파악 불가
	- 키워드 기반 검색의 장점은 빠른 속도와 (상대적으로) 쉬운 구현
	- 그러나, 복지의 경우 다양한 자격 조건(연령, 소득 수준, 성별)이 존재해 키워드 기반 검색 시 정확한 검색 결과를 얻기 힘듦(예시 -> 20대 청년 취업 지원 검색 시 결과가 존재하지 않음)
- 따라서 문맥 기반 검색(Semantic Search) 구현을 1차로, 2차적으로 생성형 AI를 이용한 답변 종합 및 정제를 목표로 하여 구현

- 연구 목적
	- Vector Store 기반 검색(Retreival) + 생성형 AI를 통한 답변 종합(Augmenting, Generating) 구현
	- 한국어, 특히 복지 도메인에 기반한 Embedding Model 구현
	- Embedding Model Domain Adaptation, Tokenizer Vocab 추가
---
### II. Pre-requirements

- Langchain(RAG Pipeline)
- OpenAI API(LLM)
	> Using gpt-3.5-turbo-1106, gpt-4-1106-preview
- Vector Store(chroma db)
- Base Embedding Model : [KoSimCSE-roberta-multitask](https://huggingface.co/BM-K/KoSimCSE-roberta-multitask)
- Streamlit
---
### III. Description

- RAG는 다음과 같은 Arcitecture로 구성되어 있습니다.

![rag](/image/rag.png)

- 세부 작동 방식은 다음과 같습니다.
	- Document Pre-Processing `workspace/mdLoader.py`
		- `.md`형식으로 미리 변경, 정제한 원본 `.pdf`파일들을 각각 불러온 후, 미리 설정해 놓은 `seperators`인자에 따라서 분리합니다.
		- 그리고 외부에서 불러와 저장 해 놓은 metadata 정보(url, tag) -> `workspace/url_table.csv`, `workspace/metadata.json`까지 각각 파일의 제목과 합쳐서 Langchain에서 이용 가능한 완전한 `Document`Object List를 생성합니다.
	- Embedding Model Loading `workspace/embeddingSetup.py`
		- 미리 저장해놓은 `sentencetransformer`기반 Embedding Model을 불러와서 이용 가능하도록 대기시킵니다.
	- Vector Database Initialize `chromaVectorStore.py`
		- 앞서 말한 전처리, Embedding Model 과정에서 생성된 Object를 Chromadb Init을 통해 불러와서 Embedding 연산을 거친 후 지정한 Local PATH에 저장하는 Wrapper 역할을 수행합니다.
	- Retrieval Augment Generating `rag.py`
		- `chromaVectorStore.py`를 한번 더 Wrapping하는 Class로, 생성된 VectorStore Object, Embedding Object, Document Object를 Langchain을 통해 이용해, 결과를 생성합니다.
	- Main `app.py`
		- 마지막으로 이 모든 결과를 Streamlit을 통해 시각적으로 보여줍니다.

- Prompt
```
You are an expert on welfare system.

Please respond to people's questions based on the following information.

Your answer should be kind, detailed, and informative, especially for those unfamiliar with the system.

The format for documents and questions is as follows:



context below here :

{context}

==============================================================

Question:

{question}

The answer must be written solely in Korean.

return answer only.
```

여러 Prompt 방식을 이용하면서, 다음과 같은 결과를 도출할 수 있었습니다.
- Simple is Best
	- 여러 복잡한 CoT(Chain of Thought), Analogical Prompting 등등 방법론을 이용하다가 내린 최종 결론
	- 결과를 비교해 보았을 때 이게 Q&A 시스템에는 적합하지 않거나, Input Token 대비 효율적인 Output에 대한 의구심이 들었다.
	- 적어도 OpenAI를 이용할 때(Mystrall처럼 Input, Output Prompt 형식이 정해져 있는 경우 말고는),  다음과 같은 Prompt가 가장 효율이 좋다고 생각했다.
		- 목적은 언제나 간단하게
		- 짧은 문장을 여러번
		- 영어를 이용해서 질문하고 답변을 한국어로 바꿔주는 형식
	