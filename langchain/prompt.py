from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate

###### form pydantic output parser
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

def _get_pydantic():
    """ pydantic 형태로 결과 parsing해서 돌려줌. -> output 구조 설정과 결과 parsing 2번 사용해야 함. """
    class analogical_thinking(BaseModel):
        reasoning: list[str] = Field(description="예시로 생각한 3가지 논리적 구조")
        answer: list[str] = Field(descripton="예제에 따라 생성한 질문")

    output = PydanticOutputParser(pydantic_object=analogical_thinking)
    return output

##### example
def da_format_prompt(context:str, num_questions_per_chunk:int)->str:

    # 전체 구조
    full_template = """
    #Problem : {instruction}
    ==========
    #Recall : {recall}
    #Explain your reasoning : {explain}
    ==========
    {answer}
    """

    # 전체 구조에서 {instruction}에 해당하는 부분. num_questions_per_chunk, context는 input variable로서 작동.
    instruction_template = """
    당신은 복지 사이트에서 복지 제도를 검색하는 사용자입니다. 복지 제도를 인터넷에서 검색 한다고 했을 때, {num_questions_per_chunk}개의 질문을 검색하는 것이 당신이 할 일입니다.
    질문들은 전체 문서에 걸쳐서 다양한 내용을 포함해야 하고, 질문 은 제공된 문맥 정보로만 제한해야 합니다.
    제공된 문맥 정보를 바탕으로, 그리고 이전 지식은 사용하지 않고 오직 정보에 기반하여 질문을 생성하십시오.
    ==========
    #Context : 
    {context}"""
    instruction_prompt = PromptTemplate.from_template(instruction_template)

    # 전체 구조에서 {recall}에 해당하는 부분 -> Analogical Prompting에서 LLM이 문서를 보고 생성한 예시
    recall_template = """문제와 관련한 예시 3개를 떠올려 보세요. 각각의 문제는 논리적인 구조로 생성해야 합니다."""
    recall_prompt = PromptTemplate.from_template(recall_template)

    # 전체 구조에서 {explain}에 해당하는 부분 -> recall에 대한 이유를 논리적 구조에 따라 생성함
    explain_template = """ 생성한 이유를 생각하고 작성해보세요. 이유의 형식은 다음과 같습니다. 
    - 첫 번째로, 저는 이렇게 생각했습니다 ... 
    - 다음으로, 저는 이렇게 생각했습니다 ...
    - 결론적으로, 저는 이렇게 생각했습니다 ... """
    explain_prompt = PromptTemplate.from_template(explain_template)

    # 전체 구조에서 {answer}에 해당하는 부분 -> 앞의 recall과 explain을 보고, LLM에서 num_questions_per_chunk의 개수에 해당하는 문제의 수만큼 돌려줌. format_instruction은 pydantic object.
    answer_template = """앞선 이유를 바탕으로, {num_questions_per_chunk}개의 질문을 생성해주세요. 다른 내용 없이 생성한 질문만 작성하면 됩니다. 질문은 무조건 한국어로 작성해주세요.
    {num_questions_per_chunk}개의 질문을 생성하면 다음과 같습니다.

    {format_instruction}
    ... """
    answer_prompt = PromptTemplate.from_template(answer_template)

    # 전체 프롬프트 구성
    input_prompts = [
        ("instruction", instruction_prompt),
        ("recall", recall_prompt),
        ("explain", explain_prompt),
        ("answer", answer_prompt),
    ]
    prompt = PromptTemplate.from_template(full_template)
    prompt_pipeline = PipelinePromptTemplate(final_prompt=prompt, pipeline_prompts=input_prompts)

    # 전체 프롬프트를 구성하고, input variable에 해당하는 변수 넣어서 LLM에 넣을 최종 내용으로 작성한다.
    pydantic = _get_pydantic()

    #prompt config
    final = prompt_pipeline.format(
        context = context,
        num_questions_per_chunk = num_questions_per_chunk,
        format_instruction = pydantic.get_format_instructions(),
    )
    return final

# print(da_format_prompt(context="안녕",num_questions_per_chunk=2))





