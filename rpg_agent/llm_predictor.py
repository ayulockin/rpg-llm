import os

import weave
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class LLMPredictor(weave.Model):
    model_name: str = "gpt-4o-mini"

    @weave.op()
    def predict(self, messages: list[dict], **kwargs):
        return llm_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs,
        )
