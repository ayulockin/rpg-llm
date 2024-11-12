import weave
from dotenv import load_dotenv

from rpg_agent.agent import ScreenshotDescriptionAgent
from rpg_agent.llm_predictor import LLMClient, LLMPredictor
from rpg_agent.control_interface import InputExecutor, KeyStroke, MouseAction


# load_dotenv()
# weave.init(project_name="ml-colabs/rpg-agent")
# agent = ScreenshotDescriptionAgent(
#     frame_predictor=LLMPredictor(model_name="gpt-4o-mini", llm_client=LLMClient.OPENAI),
# )
# agent.predict()

executor = InputExecutor(game_window_size=(1920, 1080))

# executor.execute_keystroke(KeyStroke.i) # open inventory
# import time
# time.sleep(10)  # 1 second delay to allow inventory to fully open

# executor.execute_keystroke(KeyStroke.i) # close inventory

executor.execute_mouse_action(MouseAction.mouse_left, x=700, y=500)
executor.execute_mouse_action(MouseAction.mouse_right, x=200, y=200)
executor.execute_mouse_action(MouseAction.move, x=300, y=300)
