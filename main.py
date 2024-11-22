import weave
from dotenv import load_dotenv
load_dotenv()

from rpg_agent.agents import (
    ScreenshotDescriptionAgent,
    InventoryAgent,
    StorageAgent,
    Florence2ScreenshotDetectionAgent,
    ScreenshotDescriptionAgent
)
from rpg_agent.control_interface import InputExecutor, KeyStroke, MouseAction
from rpg_agent.utils import get_bbox_center
from rpg_agent.llm_predictor import LLMPredictor


weave.init(project_name="ml-colabs/rpg-agent")


# Two agents needed by other agents
GAME_WINDOW_SIZE = (1920, 1080)
executor = InputExecutor(game_window_size=GAME_WINDOW_SIZE)
# executor.execute_mouse_action(MouseAction.mouse_left, x=1050, y=580)
screenshot_description_agent = ScreenshotDescriptionAgent()

# Microagent 1: Inventory Agent
inventory_agent = InventoryAgent()
print(inventory_agent.predict(executor, screenshot_description_agent))

# Microagent 2: Storage Agent
storage_agent = StorageAgent()
print(storage_agent.predict(executor))

# Microagent 3: Object Detection Agent
# agent = Florence2ScreenshotDetectionAgent(
#     model_name="microsoft/Florence-2-large",
#     llm=LLMPredictor(model_name="gpt-4o"),
# )
# print(agent.predict())

# Microagent 4: Screenshot Description Agent
# screenshot_description_agent = ScreenshotDescriptionAgent(
#     instruction="""
#     We are playing Divinity: Original Sin 2 game. You are provided the current frame of the game. Your job is to first analyze the frame and then describe the contents of the frame in great detail. You need to find the objects that might be interactable in the game.
#     We have overlayed a grid of where each cell is 100x100 pixels on the frame. For the objects of interest, you need to tell which cell of the grid they are in.
#     Return the list of objects of interest and the grid coordinates of each object.

#     The horizontal axis or the x-axis is represented by alphabets from A to S and the vertical axis or the y-axis is represented by numbers from 1 to 11.
#     """.strip()
# )
# print(screenshot_description_agent.predict())
