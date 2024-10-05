from enum import Enum
from typing import Optional, Union

import instructor
import weave
from instructor import Instructor
from mistralai import Mistral
from openai import OpenAI
from PIL import Image

from .utils import encode_image


class LLMClient(str, Enum):
    OPENAI = "openai"
    MISTRAL = "mistral"


class LLMPredictor(weave.Model):
    model_name: str
    llm_client: LLMClient
    _llm_client: Union[OpenAI, Mistral] = None
    _structured_llm_client: "Instructor" = None

    def __init__(self, model_name: str, llm_client: LLMClient):
        super().__init__(model_name=model_name, llm_client=llm_client)
        if llm_client == LLMClient.OPENAI:
            self._llm_client = OpenAI()
            self._structured_llm_client = instructor.from_openai(self._llm_client)
        elif llm_client == LLMClient.MISTRAL:
            self._llm_client = Mistral()
            self._structured_llm_client = instructor.from_mistral(self._llm_client)

    @weave.op()
    def frame_messages(
        self,
        system_prompt: Optional[str] = None,
        user_prompts: Union[str, list[str]] = [],
    ):
        messages = []

        if self.llm_client == LLMClient.OPENAI:
            messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
            for idx, user_prompt in enumerate(user_prompts):
                if isinstance(user_prompt, Image.Image):
                    user_prompts[idx] = {
                        "type": "image_url",
                        "image_url": {"url": encode_image(user_prompt)},
                    }
                elif isinstance(user_prompt, str):
                    user_prompts[idx] = {"type": "text", "text": user_prompt}
                else:
                    raise ValueError(f"Invalid user prompt type: {type(user_prompt)}")
            messages.append({"role": "user", "content": user_prompts})

        return messages

    @weave.op()
    def predict(
        self,
        system_prompt: Optional[str] = None,
        user_prompts: Union[str, list[str]] = [],
        **kwargs,
    ):
        if self.llm_client == LLMClient.OPENAI:
            if "response_model" in kwargs:
                return self._structured_llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=self.frame_messages(system_prompt, user_prompts),
                    **kwargs,
                )
            elif "response_format" in kwargs:
                return (
                    self._llm_client.beta.chat.completions.create(
                        model=self.model_name,
                        messages=self.frame_messages(system_prompt, user_prompts),
                        **kwargs,
                    )
                    .choices[0]
                    .message.parsed
                )
            else:
                return (
                    self._llm_client.chat.completions.create(
                        model=self.model_name,
                        messages=self.frame_messages(system_prompt, user_prompts),
                        **kwargs,
                    )
                    .choices[0]
                    .message.content
                )
        else:
            raise ValueError(f"Invalid LLM client: {self.llm_client}")
