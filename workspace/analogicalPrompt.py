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

## test 
if __name__ == "__main__" :
    print(generateAnalogicalPrompt())
