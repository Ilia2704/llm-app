import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import os
os.environ["OPENAI_API_KEY"] = "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"

from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.embeddings import OpenAIEmbeddings
import openai

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
openai_client = openai.OpenAI()
evaluator_embeddings = OpenAIEmbeddings(client=openai_client)

from ragas import SingleTurnSample
from ragas.metrics import AspectCritic

test_data = {
    "user_input": "summarise given text\nThe company reported an 8% rise in Q3 2024, driven by strong performance in the Asian market. Sales in this region have significantly contributed to the overall growth. Analysts attribute this success to strategic marketing and product localization. The positive trend in the Asian market is expected to continue into the next quarter.",
    "response": "The company experienced an 8% increase in Q3 2024, largely due to effective marketing strategies and product adaptation, with expectations of continued growth in the coming quarter.",
}



metric = AspectCritic(name="summary_accuracy",llm=evaluator_llm, definition="Verify if the summary is accurate.")
test_data = SingleTurnSample(**test_data)


from datasets import load_dataset
from ragas import EvaluationDataset
eval_dataset = load_dataset("explodinggradients/earning_report_summary",split="train")
eval_dataset = EvaluationDataset.from_hf_dataset(eval_dataset)
print("Features in dataset:", eval_dataset.features())
print("Total samples in dataset:", len(eval_dataset))