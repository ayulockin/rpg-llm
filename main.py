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
    InventoryAgent,
    StorageAgent,
)
from rpg_agent.llm_predictor import LLMPredictor
from rpg_agent.control_interface import InputExecutor, KeyStroke, MouseAction
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


# inventory_agent = InventoryAgent()
# inventory_agent.predict(
#     executor=executor,
#     llm_frame_description=llm_frame_description
# )

# storage_agent = StorageAgent()
# storage_agent.predict()
