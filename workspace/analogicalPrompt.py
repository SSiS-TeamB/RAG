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
    
    The format for documents and questions is as follows:
    
    context below here :
    'point 1'
    {context}
    'point 2'
    ==============================================================
    Question:
    {question}

    ==============================================================
    
    documents are between 'point 1' and 'point 2'. And there are '\n***\n' string between documents
    title of document appears in front of '내용'
    
    First, choose documents relevant with question. This procedure is especially important.
    Now I call refered documents 'relevant docs'.
    Next, generate answer based on relevant docs.
    The answer should be kind, detailed, comprehensively, and informative, especially for those unfamiliar with the system.
    Now I call this generated answer 'your answer'. 
    And 'titles of r_doc' are titles of each relevant docs.

    If there is only one or no relevant docs, should tell me kindly and detailedly the reason why you refer to the relevant docs and doesn't refer to other documents.
    Now I call this reason 'choose reason'.

    Please keep in this format '[your answer]\n\n참고 문서: [elements of titles of r_doc]\n\n<참고> [choose reason(if exists)]'
    
    remember next rule
    1. only if there is one or on relevant docs, tell me the choose reason.
    2. don't contain choose reason and titles of r_doc and relevant docs in your answer.
    3. your answer, elements of titles of r_doc, and choose reason must be written solely in Korean.

    """)
    # ==============================================================
    # The answer must be written solely in Korean.
    # return answer only.""")
    return prompt_template

## test 
if __name__ == "__main__" :
    print(get_normal_prompt())
