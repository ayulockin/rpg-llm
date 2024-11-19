import weave
from dotenv import load_dotenv
load_dotenv()

from rpg_agent.agents import (
    Agent,
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
screenshot_description_agent = ScreenshotDescriptionAgent()

# Microagent 1: Inventory Agent
inventory_agent = InventoryAgent()
# print(inventory_agent.predict(executor, screenshot_description_agent))

# Microagent 2: Storage Agent
storage_agent = StorageAgent()
print(storage_agent.predict(executor))

# executor.execute_mouse_action(MouseAction.mouse_left, x=1100, y=540)

# agent = Florence2ScreenshotDetectionAgent(
#     model_name="microsoft/Florence-2-large",
#     llm=LLMPredictor(model_name="gpt-4o"),
# )
# agent.predict()
