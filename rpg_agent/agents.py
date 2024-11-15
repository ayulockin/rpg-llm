import weave
import json
import cv2
import numpy as np
import random
from PIL import ImageGrab, Image, ImageDraw
import pyautogui
import time
import os
from abc import abstractmethod

from openai.types.chat.chat_completion_message_tool_call import Function

from .llm_predictor import LLMPredictor
from .tools import inventory_agent_tools
from .control_interface import (
    KeyStroke, KEYSTROKE_STRING_MAPPING, 
    InputExecutor, 
    MouseAction, MOUSE_ACTION_STRING_MAPPING
)
from .utils import *


class Agent(weave.Model):
    llm: LLMPredictor = LLMPredictor()
    name: str = "base_agent"
    model_name: str = "gpt-4o-mini"
    instruction: str = "You are a helpful agent that can perform actions in the game."
    tools: list[dict] = []
    chat_history: list[dict] = []

    def model_post_init(self, __context):
        self.llm = LLMPredictor(model_name=self.model_name)

    @abstractmethod
    @weave.op()
    def predict(self):
        pass


class ScreenshotDescriptionAgent(Agent):
    name: str = "screenshot_description_agent"
    model_name: str = "gpt-4o"
    instruction: str = """
    We are playing Divinity: Original Sin 2 game. You are provided the current frame of the game.
    We are trying to play this game autonomously thus we need to know what the current state of the game is. Your task is to describe the frame in great details. Use bullet points and be very specific.
    
    - We want to know the objects of interest in the frame. These objects are the ones that we might be able to interact with in the Divinity: Original Sin 2 game.
    - We also want to know the position of the main player or character in the frame. This will help us to know where we are in the game.
    - Tell the count of the objects of interest if there are multiple objects of the same type where necessary.
    - Describe the position of the objects of interest with respect to the player or character.
    - Also describe how far the object of interest is from the player or character.
    - We also want to know your opinion on what should be the best course of action given the current state of the game.

    Be detailed with your description.
    """.strip()

    @weave.op()
    def predict(self):
        image = get_game_window()

        response = self.llm.predict(
            messages=[
                {
                    "role": "system", 
                    "content": self.instruction
                },
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encode_image(image)
                            }
                        }
                    ]
                },
            ],
        )

        return {
            "game_frame": image,
            "prediction": response.choices[0].message.content,
        }


class InventoryAgent(Agent):
    name: str = "inventory_agent"
    model_name: str = "gpt-4o"
    instruction: str = """
    You are responsible for opening the inventory and describing the contents. You have to do the following in order. STRICTLY follow the order.

    1. For opening the inventory, use the key 'i'. You have access to the `execute_keystroke` method. Once the inventory is open, you can describe the contents.

    2. For describing the inventory, use the `screenshot_description_agent` function. This function takes in no arguments. Once you have the description, you can close the inventory.

    3. For closing the inventory, use the key 'i' again. You have access to the `execute_keystroke` method

    4. Once this task is complete, you should return JUST a valid boolean value: False.
    """
    tools: list[dict] = inventory_agent_tools
    chat_history: list[str] = [
        {"role": "system", "content": instruction}
    ]

    @weave.op()
    def predict(
        self,
        executor: InputExecutor,
        screenshot_description_agent: ScreenshotDescriptionAgent
    ):
        while True:
            response = self.llm.predict(
                messages=self.chat_history,
                tools=self.tools,
            )

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    if tool_call.function.arguments:
                        arguments = json.loads(tool_call.function.arguments)

                    if tool_call.function.name == "execute_keystroke":
                        executor.execute_keystroke(
                            KeyStroke(KEYSTROKE_STRING_MAPPING[arguments["keystroke"]])
                        )
                        self.chat_history.append(
                            {"role": "assistant", "content": "Pressed key: " + arguments["keystroke"]}
                        )
                    elif tool_call.function.name == "screenshot_description_agent":
                        inventory_description = screenshot_description_agent.predict()["prediction"]
                        self.chat_history.append(
                            {"role": "assistant", "content": f"We gathered the description of the inventory: {inventory_description}"}
                        )

            if response.choices[0].message.content == "False":
                break

        return self.chat_history


"""
storage agent is responsible for picking the items from the storage units and putting them in the inventory.

1. get the coordinates for the storage unit.
2. click on the storage unit.
3. if the storage unit is open, use the `llm_frame_description` function to get the description of the storage unit. like how many items are there, what are the items.
4. click on the take all button.
5. click on the close button.
6. repeat the process for the next storage unit.
"""


@weave.op()
def get_coordinates() -> tuple[int, int]:
    return 1200, 560

storage_agent_tools = [
    {
        "type": "function",
        "function": {
            "name": "screenshot_description_agent",
            "description": "Call this whenever you need to know the current frame of the game. This function takes in no arguments.",
            "parameters": {},
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_mouse_action",
            "description": "Call this whenever you need to move the mouse. This function takes in a `MouseAction` enum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mouse_action": {
                        "type": "string",
                        "enum": ["mouse_left"],
                    }
                },
                "required": ["mouse_action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_coordinates",
            "description": "Call this whenever you need to get the coordinates for the storage unit. This function takes in no arguments.",
            "parameters": {},
        }
    }
]


class StorageAgent(Agent):
    name: str = "storage_agent"
    model_name: str = "gpt-4o"
    tools: list[dict] = storage_agent_tools
    instruction: str = """
    You are responsible for picking the items from the storage units and putting them in the inventory. You have to do the following in order. STRICTLY follow the order.

    1. First get the coordinates for the storage unit. You have access to the `get_coordinates` function. Once you have the coordinates, you can click on the storage unit.

    2. Now click on the storage unit. You have access to the `execute_mouse_action` function.

    3. Assuming the storage unit is open, use the `frame_description` function to get the description of the storage unit - like how many items are there, what are the items.

    4. Now we will take all the items from the storage unit. Click on the take all button. You have access to the `execute_mouse_action` function.

    5. Now click on the close button. You have access to the `execute_mouse_action` function.

    6. Once this task is complete, you should return JUST a valid boolean value: False.

    Make sure to do one task at a time.
    """.strip()

    screenshot_description_prompt: str = """
    You are playing Divinity: Original Sin 2. You are given the frame of the game with the storage unit opened. Please describe the contents of the storage unit in great detail. The storag unit will have a checkboard pattern where each cell will have an item. Tell me the total count of the items and the count of each item.
    """.strip()
    screenshot_description_agent: ScreenshotDescriptionAgent = ScreenshotDescriptionAgent(instruction=screenshot_description_prompt)

    chat_history: list[str] = [
        {"role": "system", "content": instruction}
    ]

    @weave.op()
    def predict(
        self,
        executor: InputExecutor,
    ):
        max_iterations = 10
        while True:
            response = self.llm.predict(
                messages=self.chat_history,
                tools=self.tools,
            )
            print("The response is: ", response)

            if response.choices[0].message.tool_calls:
                print("The tool calls are: ", response.choices[0].message.tool_calls)
                for tool_call in response.choices[0].message.tool_calls:
                    if tool_call.function.arguments:
                        arguments = json.loads(tool_call.function.arguments)
                        print("The arguments are: ", arguments)

                    if tool_call.function.name == "get_coordinates":
                        print("Getting the coordinates")
                        coordinates = get_coordinates()
                        self.chat_history.append(
                            {"role": "assistant", "content": f"Got the coordinates: {coordinates}"}
                        )

                    if tool_call.function.name == "execute_mouse_action":
                        print("Executing the mouse action")
                        executor.execute_mouse_action(
                            MouseAction(MOUSE_ACTION_STRING_MAPPING[arguments["mouse_action"]]),
                            coordinates[0],
                            coordinates[1],
                        )
                        self.chat_history.append(
                            {"role": "assistant", "content": f"Clicked on the storage unit"}
                        )

                    if tool_call.function.name == "screenshot_description_agent":
                        print("Taking the screenshot description")
                        storage_unit_description = self.screenshot_description_agent.predict()["prediction"]
                        self.chat_history.append(
                            {"role": "assistant", "content": f"We gathered the description of the storage unit: {storage_unit_description}"}
                        )

                    # TODO: handle taking the items and closing the storage unit

            if response.choices[0].message.content == "False":
                break

            max_iterations -= 1
            if max_iterations == 0:
                break

        return self.chat_history
