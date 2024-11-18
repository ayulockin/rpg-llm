import json
import time
from abc import abstractmethod
from typing import Optional

import cv2
import numpy as np
import pyautogui
import weave
from openai.types.chat.chat_completion_message_tool_call import Function
from PIL import Image, ImageGrab

from .control_interface import KEYSTROKE_STRING_MAPPING, InputExecutor, KeyStroke
from .llm_predictor import LLMPredictor
from .models import (
    Florence2DetectionModel,
    Owlv2DetectionModel,
    UltralyticsDetectionModel,
)
from .tools import (
    inventory_agent_tools, storage_agent_tools
)
from .utils import (
    draw_bbox,
    encode_image,
    get_game_window,
    parse_json,
    save_combined_image,
    get_template_match,
)
from .control_interface import (
    KeyStroke, KEYSTROKE_STRING_MAPPING, 
    InputExecutor, 
    MouseAction, MOUSE_ACTION_STRING_MAPPING
)


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
    )

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
                    ],
                },
            ],
        )

        return {
            "game_frame": image,
            "prediction": response.choices[0].message.content,
        }


class WhereIsTheCharacterAgent(weave.Model):
    object_detector: LLMPredictor  # it can also be a standard object detection model
    prompt_task: str = (
        """We need to detect the main character in the frame of the game. We are playing Divinity: Original Sin 2. Please carefully examine the frame and detect the main character with precise bounding boxes that closely enclose the character. 

    The frame is 1920x1080 (width, height) with the main character likely to be centered in the screen. The bounding boxes should be non-normalized (pixel values). The detection should be precise and accurate because we will use the detected position to control the character.

    Other than the image itself, we are also providing the description of the frame to help you detect the objects better.

    Frame description: {frame_description}
    """.strip()
    )

    prompt_instructions: str = (
        """
    Return ONLY a JSON object containing the bounding boxes. 
        
    Example output:
    {
        "element": "main_character", 
        "bbox": [x_min, y_min, x_max, y_max],
        "confidence": 0.95
    }
    """.strip()
    )

    @weave.op()
    def predict(self, frame_description: str):
        image = get_game_window()  # utils.py

        system_prompt = self.prompt_task.format(frame_description=frame_description)
        system_prompt += "\n" + self.prompt_instructions

        print(system_prompt)

        response = self.object_detector.predict(
            system_prompt=system_prompt,
            user_prompts=[image],
        )

        try:
            bboxes = parse_json(response.choices[0].message.content)
            print(bboxes)
            output = {
                "game_frame": image,
                "prediction_image": draw_bbox(image, bboxes),  # utils.py
                "prediction": bboxes,
            }
        except json.JSONDecodeError:
            output = {
                "game_frame": image,
                "error": str(response.choices[0].message.content),
            }

        return output


class HoverDetectionAgent(weave.Model):
    save_dir: str = "detected_changes"
    crop_size: int = 80
    grid_step: int = 50
    contour_threshold: int = 100
    exclusion_size: int = 200
    search_width: int = 500
    search_height: int = 500

    @weave.op()
    def hover_and_detect_around_character(self, character_x, character_y):
        """Hover mouse in a square area around the character, excluding a central region, and detect changes."""
        detected_points = []
        detection_count = 0  # Counter for naming saved images

        # Define the bounds of the search area
        x_min = max(0, character_x - self.search_width // 2)
        x_max = min(pyautogui.size().width, character_x + self.search_width // 2)
        y_min = max(0, character_y - self.search_height // 2)
        y_max = min(pyautogui.size().height, character_y + self.search_height // 2)

        # Loop through points within the defined square search area
        for x in range(x_min, x_max, self.grid_step):
            for y in range(y_min, y_max, self.grid_step):
                # Skip points within the exclusion zone around the character
                if (
                    abs(x - character_x) < self.exclusion_size // 2
                    and abs(y - character_y) < self.exclusion_size // 2
                ):
                    continue

                # Move mouse to the (x, y) position
                pyautogui.moveTo(x, y)
                time.sleep(0.1)  # Small delay to allow the screen to update

                # Capture a larger region around the pointer (e.g., 80x80 pixels)
                region = (
                    x - self.crop_size // 2,
                    y - self.crop_size // 2,
                    x + self.crop_size // 2,
                    y + self.crop_size // 2,
                )
                current_frame = ImageGrab.grab(bbox=region)
                current_frame = cv2.cvtColor(np.array(current_frame), cv2.COLOR_RGB2BGR)

                # Apply edge detection to emphasize highlighted regions
                current_edges = cv2.Canny(current_frame, 50, 150)

                # If we have a previous frame, compare it with the current frame
                if "previous_edges" in locals():
                    # Calculate absolute difference between edge frames
                    diff = cv2.absdiff(previous_edges, current_edges)

                    # Find contours of the changes
                    contours, _ = cv2.findContours(
                        diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    # Filter contours by area to reduce noise
                    significant_change = False
                    for contour in contours:
                        if cv2.contourArea(contour) > self.contour_threshold:
                            significant_change = True
                            break

                    # If a significant contour is found, record this point
                    if significant_change:
                        detected_points.append((x, y))
                        print(f"Significant change detected at ({x}, {y})")

                        # Annotate and save the before and after images side by side
                        combined_image = save_combined_image(
                            previous_frame,
                            current_frame,
                            x,
                            y,
                            detection_count,
                            self.save_dir,
                        )
                        detection_count += 1  # Increment the detection count

                # Update the previous frame edges and current frame
                previous_edges = current_edges
                previous_frame = current_frame

        return detected_points


class PlannerAgent(weave.Model):
    @weave.op()
    def predict(self):
        # will return the tool name
        return Function(
            name="execute_mouse_action",
            arguments="mouse_action=MouseAction.mouse_left, x=1100, y=600",
        )
      

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
        },
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

    4. Now we will take all the items from the storage unit. Click on the take all button. To do this you have access to the `get_template_match` function to get the coordinates of the take all button. Once you have the coordinates, you have access to the `execute_mouse_action` function to click on the take all button.

    5. Now click on the close button. To do this you have access to the `get_template_match` function to get the coordinates of the close button. Once you have the coordinates, you have access to the `execute_mouse_action` function to click on the close button.

    6. Once this task is complete, you should return JUST a valid boolean value: False.

    Make sure to do one task at a time.
    """.strip()
    )
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

                    # TODO: what is the downstream action for this?
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


class OWLScreenshotDetectionAgent(Agent):
    object_detector: Owlv2DetectionModel = Owlv2DetectionModel()

    @weave.op()
    def predict(self, prompts: list[list[str]]):
        image = get_game_window(use_image_grab=False, monitor_index=2)
        response = self.object_detector.predict(prompts=prompts, image=image)
        return {
            "game_frame": image,
            "prediction": response,
        }


class YOLOScreenshotDetectionAgent(Agent):
    object_detector: Optional[UltralyticsDetectionModel] = None
    model_name: str = "yolo11n"

    def model_post_init(self, __context):
        self.object_detector = UltralyticsDetectionModel(model_name=self.model_name)

    @weave.op()
    def predict(self):
        image = get_game_window(use_image_grab=False, monitor_index=2)
        response = self.object_detector.predict(image=image)
        return {
            "game_frame": image,
            "prediction": response,
        }


class Florence2ScreenshotDetectionAgent(Agent):
    object_detector: Optional[Florence2DetectionModel] = None
    model_name: str = "microsoft/Florence-2-large"
    task_prompt: str = "<DENSE_REGION_CAPTION>"

    def model_post_init(self, __context):
        self.object_detector = Florence2DetectionModel(
            model_name=self.model_name, task_prompt=self.task_prompt
        )

    @weave.op()
    def describe_screenshot(self, image: Image.Image) -> list[str]:
        response = self.llm.predict(
            messages=[
                {
                    "role": "system",
                    "content": """
You are given a screenshot from the role-playing game Divinity: Original Sin 2 game.
You are suppossed to first analyze the screenshot and then describe the contents of the screenshot in great detail.
You are only suppossed give the objects present in the screenshot in a comma separated manner and nothing else.
""",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": encode_image(image)},
                        }
                    ],
                },
            ],
        )
        return response.choices[0].message.content

    @weave.op()
    def predict(self):
        image = get_game_window(use_image_grab=False, monitor_index=2)
        image_description = self.describe_screenshot(image)
        response = self.object_detector.predict(image=image, prompt=image_description)
        return {
            "game_frame": image,
            "prediction": response,
        }
 