""" 1116 12:54 의존성 문제 해결 """

from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts import PromptTemplate

#extract text
def _extract_text(file_path:str) -> str:
    with open(file_path, "r+", encoding="utf-8") as file :
        return file.read()

def generateAnalogicalPrompt() -> PipelinePromptTemplate:
    """ get text template from /prompt folder to PipelinePromptTemplate. """

    #get text from file
    prompt_wrapper = _extract_text("workspace/prompt/wrapper.txt")
    instruction_template = _extract_text("workspace/prompt/instruction.txt")
    recall_template = _extract_text("workspace/prompt/recall.txt")
    answer_template = _extract_text("workspace/prompt/answer.txt")

    #### merge prompt templates -> to Prompt Pipeline
    instruction_prompt = PromptTemplate.from_template(instruction_template)
    recall_prompt = PromptTemplate.from_template(recall_template)
    answer_prompt = PromptTemplate.from_template(answer_template)

    input_prompts = [
            ("instruction", instruction_prompt),
            ("recall", recall_prompt),
            ("answer", answer_prompt),
        ]

    final_prompt = PromptTemplate.from_template(prompt_wrapper)
    prompt_pipeline = PipelinePromptTemplate(final_prompt=final_prompt, pipeline_prompts=input_prompts)

    return prompt_pipeline

def get_normal_prompt() -> PromptTemplate:
    """ get normal RAG template -> context : query(question). """
    prompt_template = PromptTemplate.from_template("""
    You are an expert on welfare system. 
    Please respond to people's questions based on the following information. 
    Your answer should be kind, detailed, and informative, especially for those unfamiliar with the system.
    And I recommend to refer to many documents and answer comprehensively.
    The format for documents and questions is as follows:

    context below here :
    'point 1'
    {context}
    'point 2'
    ==============================================================
    Question:
    {question}

    ==============================================================
    documents are between 'point 1' and 'point 2'. title appears before '내용'.
                                                                                            
    The answer must be written solely in Korean.
    return answer and titles of all documents.
    in any case, must return anwer and titles at the bottom. in format 'answer \n\n 참고 문서: [title1, titles2,,,]'.
    if you refer to only one document, should tell me the reason why you refer to only the document and doesn't refer to other documents.
    (remember! tell me the reason only if you refer to only one document.)
    
    """)
    # ==============================================================
    # The answer must be written solely in Korean.
    # return answer only.""")
    return prompt_template

## test 
if __name__ == "__main__" :
    print(get_normal_prompt())
