""" LangChain version QA generation """

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.pydantic_v1 import BaseModel
from langchain_experimental.tabular_synthetic_data.base import SyntheticDataGenerator
from langchain_experimental.tabular_synthetic_data.openai import (
    create_openai_data_generator,
    OPENAI_TEMPLATE,
)
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_SUFFIX,
    SYNTHETIC_FEW_SHOT_PREFIX,
)

class WelfareDataset(BaseModel) :
    query = list[str] 