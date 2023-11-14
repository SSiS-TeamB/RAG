""" Analogical Prompting -> 이거 논리적 추론으로 하는건데 Retrieval 할 때도 괜찮을까? """

from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts import PromptTemplate

def generator_prompt():
    prompt_template = """
    #Instruction : 
    {instruction}
    ==========
    #Relevant Problems : 
    {recall}

    #Explain your reasoning : 
    {explain}
    ==========
    #Solve the Initial Question : {answer}

    """

    instruction_template = """
    "당신은 대한민국의 사회복지 전문가로서, 다양한 배경과 요구를 가진 사람들을 지원합니다. 
    여러 사회 문제에 대한 깊은 이해를 바탕으로, 각 개인의 상황에 맞는 도움을 제공해야 합니다. 
    제공된 #context만을 사용하여, 사람들의 #Initial Question에 대해 상세하고 정확한 답변을 하세요."

    #context : {context}

    #Initial Question : {question}

    """


    recall_template = """ 정보와 관련된 세 가지 사회 복지 문제를 떠올려 보세요. 각각의 문제에 대해 :
        
        - 관련된 문제는 다음과 같이 작성해야 합니다. "Q: "
        - 문제와 해결책에 대한 논리적인 추론은 다음과 같이 작성해야 합니다.  "A: " 

    """

    explain_template = """ 문제를 생성한 이유를 설명해보세요. 이유의 형식은 다음과 같습니다
    - 첫 번째로, 저는 이렇게 생각했습니다 ... 
    - 다음으로, 저는 이렇게 생각했습니다 ...
    - 결론적으로, 저는 이렇게 생각했습니다 ... 
    """

    answer_template = """ 따라서 #Initial Question의 해결책은 다음과 같습니다. \n 
    "Q: {question}"
    "A: ... " 

    """

    #### merge prompt templates -> to Prompt Pipeline
    instruction_prompt = PromptTemplate.from_template(instruction_template)
    recall_prompt = PromptTemplate.from_template(recall_template)
    explain_prompt = PromptTemplate.from_template(explain_template)
    answer_prompt = PromptTemplate.from_template(answer_template)

    input_prompts = [
            ("instruction", instruction_prompt),
            ("recall", recall_prompt),
            ("explain", explain_prompt),
            ("answer", answer_prompt),
        ]

    final_prompt = PromptTemplate.from_template(prompt_template)
    prompt_pipeline = PipelinePromptTemplate(final_prompt=final_prompt, pipeline_prompts=input_prompts)

    return prompt_pipeline

