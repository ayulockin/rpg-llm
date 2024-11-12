import weave
from dotenv import load_dotenv
load_dotenv()

from rpg_agent.agent import ScreenshotDescriptionAgent, ObjectDetectionAgent
from rpg_agent.llm_predictor import LLMClient, LLMPredictor
from rpg_agent.control_interface import InputExecutor, KeyStroke, MouseAction

weave.init(project_name="ml-colabs/rpg-agent")

frame_description = ScreenshotDescriptionAgent(
    frame_predictor=LLMPredictor(model_name="gpt-4o-mini", llm_client=LLMClient.OPENAI),
)
frame_description = frame_description.predict()["prediction"]
print(frame_description)

agent = ObjectDetectionAgent(
    object_detector=LLMPredictor(model_name="gpt-4o", llm_client=LLMClient.OPENAI),
)

agent.predict(frame_description=frame_description)

# executor = InputExecutor(game_window_size=(1920, 1080))

# # executor.execute_keystroke(KeyStroke.i) # open inventory
# # import time
# # time.sleep(10)  # 1 second delay to allow inventory to fully open

# # executor.execute_keystroke(KeyStroke.i) # close inventory

# executor.execute_mouse_action(MouseAction.mouse_left, x=700, y=500)
# executor.execute_mouse_action(MouseAction.mouse_right, x=200, y=200)
# executor.execute_mouse_action(MouseAction.move, x=300, y=300)
