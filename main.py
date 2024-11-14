import weave
import json
import time
from dotenv import load_dotenv
load_dotenv()

from rpg_agent.agent import (
    ScreenshotDescriptionAgent,
    WhereIsTheCharacterAgent,
    HoverDetectionAgent,
    PlannerAgent,
)
from rpg_agent.llm_predictor import LLMPredictor
from rpg_agent.control_interface import InputExecutor, KeyStroke, MouseAction, KEYSTROKE_STRING_MAPPING
from rpg_agent.utils import get_bbox_center
from openai.types.chat.chat_completion_message_tool_call import Function

weave.init(project_name="ml-colabs/rpg-agent")


GAME_WINDOW_SIZE = (1920, 1080)
executor = InputExecutor(game_window_size=GAME_WINDOW_SIZE)

llm_frame_description = ScreenshotDescriptionAgent(
    llm=LLMPredictor(model_name="gpt-4o-mini"),
)

find_character_in_frame = WhereIsTheCharacterAgent(
    object_detector=LLMPredictor(model_name="gpt-4o"),
)

hover_detection = HoverDetectionAgent()


class GameAgent(weave.Model):    
    @weave.op()
    def run(self):
        frame_description = llm_frame_description.predict()["prediction"]
        print(frame_description)

        bbox_data = find_character_in_frame.predict(frame_description=frame_description)
        print(bbox_data)

        bbox_center = get_bbox_center(bbox_data["prediction"]["bbox"])
        print(bbox_center)

        detected_points = hover_detection.hover_and_detect_around_character(bbox_center[0], bbox_center[1])
        print("Detected points:", detected_points)

        for d in detected_points:
            executor.execute_mouse_action(MouseAction.mouse_left, x=d[0], y=d[1])
            time.sleep(0.1)


class GameAgent(weave.Model):
    @weave.op()
    def run(self):
        # describer agent
        frame_description = llm_frame_description.predict()["prediction"]
        print(frame_description)

        # planner agent
        planner_agent = PlannerAgent()
        plan = planner_agent.predict()

        # Checking if the plan is a tool use. 
        # If yes, proceed to execute the tool use.
        # If not, TODO: add fallback logic
        if isinstance(plan, Function):
            function_name = plan.name
            function_arguments = plan.arguments

            # TODO: argument parsers?
            args_dict = {}
            for arg in function_arguments.split(", "):
                key, value = arg.split("=")
                if key == "mouse_action":
                    args_dict[key] = eval(value)  # Evaluate MouseAction enum
                else:
                    args_dict[key] = int(value)  # Convert x,y to integers

            # TODO: tool executor?
            if function_name == "execute_mouse_action":
                executor.execute_mouse_action(**args_dict)


# game_agent = GameAgent()
# game_agent.run()


"""
1. Press i
2. Grab screen
3. Describe screen
4. Prss i
"""

inventory_agent_tools = [
    {
        "type": "function",
        "function": {
            "name": "llm_frame_description",
            "description": "Call this whenever you need to know the current frame of the game. This function takes in no arguments.",
            "parameters": {},
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_keystroke",
            "description": "Call this whenever you need to press a key. This function takes in a `KeyStroke` enum.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keystroke": {
                        "type": "string",
                        "enum": list(KEYSTROKE_STRING_MAPPING.keys()),
                    }
                },
                "required": ["keystroke"],
            },
        },
    },
]



class InventoryAgent(weave.Model):
    llm: LLMPredictor = LLMPredictor(model_name="gpt-4o")

    agent_instruction: str = """
    You are responsible for opening the inventory and describing the contents. You have to do the following in order. STRICTLY follow the order.

    1. For opening the inventory, use the key 'i'. You have access to the `execute_keystroke` method. Once the inventory is open, you can describe the contents.

    2. For describing the inventory, use the `llm_frame_description` function. This function takes in no arguments. Once you have the description, you can close the inventory.

    3. For closing the inventory, use the key 'i' again. You have access to the `execute_keystroke` method

    4. Once this task is complete, you should return JUST a valid boolean value: False.
    """
    tools: list[dict] = inventory_agent_tools
    chat_history: list[str] = [
        {"role": "system", "content": agent_instruction}
    ]

    @weave.op()
    def predict(self):

        while True:
            response = self.llm.predict(
                messages=self.chat_history,
                tools=self.tools,
            )
            print(response)

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    print(tool_call)
                if tool_call.function.arguments:
                    arguments = json.loads(tool_call.function.arguments)

                if tool_call.function.name == "execute_keystroke":
                    executor.execute_keystroke(
                        KeyStroke(KEYSTROKE_STRING_MAPPING[arguments["keystroke"]])
                    )
                    self.chat_history.append(
                        {"role": "assistant", "content": "Pressed key: " + arguments["keystroke"]}
                    )
                elif tool_call.function.name == "llm_frame_description":
                    inventory_description = llm_frame_description.predict()
                    self.chat_history.append(
                        {"role": "assistant", "content": "We gathered the description of the inventory."}
                    )
            
            if response.choices[0].message.content == "False":
                break

        return inventory_description


inventory_agent = InventoryAgent()
inventory_agent.predict()

# executor.execute_keystroke(KeyStroke.i)

