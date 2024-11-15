import weave
import json
import time
from dotenv import load_dotenv
load_dotenv()

from rpg_agent.agents import (
    Agent,
    SimpleAgent,
    ScreenshotDescriptionAgent,
    WhereIsTheCharacterAgent,
    HoverDetectionAgent,
    PlannerAgent,
    InventoryAgent,
    StorageAgent,
)
from rpg_agent.llm_predictor import LLMPredictor
from rpg_agent.control_interface import InputExecutor, KeyStroke, MouseAction
from rpg_agent.utils import get_bbox_center
from openai.types.chat.chat_completion_message_tool_call import Function

# weave.init(project_name="ml-colabs/rpg-agent")


# GAME_WINDOW_SIZE = (1920, 1080)
# executor = InputExecutor(game_window_size=GAME_WINDOW_SIZE)


from typing import List
from pydantic import BaseModel
from abc import abstractmethod

class MyModel(BaseModel):
    foo: List[str]

    def model_post_init(self, __context):
        self.foo = [s.replace("-", "_") for s in self.foo]

my_object = MyModel(foo=["hello-there"])

print(my_object)


simple_agent = SimpleAgent()
print(simple_agent.predict())
