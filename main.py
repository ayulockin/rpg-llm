import weave
import time
from dotenv import load_dotenv
load_dotenv()

from rpg_agent.agent import ScreenshotDescriptionAgent, WhereIsTheCharacterAgent, HoverDetectionAgent
from rpg_agent.llm_predictor import LLMClient, LLMPredictor
from rpg_agent.control_interface import InputExecutor, KeyStroke, MouseAction
from rpg_agent.utils import get_bbox_center

weave.init(project_name="ml-colabs/rpg-agent")


llm_frame_description = ScreenshotDescriptionAgent(
    frame_predictor=LLMPredictor(model_name="gpt-4o-mini", llm_client=LLMClient.OPENAI),
)

fine_character_in_frame = WhereIsTheCharacterAgent(
    object_detector=LLMPredictor(model_name="gpt-4o", llm_client=LLMClient.OPENAI),
)

hover_detection = HoverDetectionAgent()


class GameAgent(weave.Model):
    game_window_size: tuple = (1920, 1080)
    
    @weave.op()
    def run(self):
        executor = InputExecutor(game_window_size=self.game_window_size)
    
        frame_description = llm_frame_description.predict()["prediction"]
        print(frame_description)

        bbox_data = fine_character_in_frame.predict(frame_description=frame_description)
        print(bbox_data)

        bbox_center = get_bbox_center(bbox_data["prediction"]["bbox"])
        print(bbox_center)

        detected_points = hover_detection.hover_and_detect_around_character(bbox_center[0], bbox_center[1])
        print("Detected points:", detected_points)

        for d in detected_points:
            executor.execute_mouse_action(MouseAction.mouse_left, x=d[0], y=d[1])
            time.sleep(0.1)

# dps = [(610, 405), (710, 305), (710, 355), (810, 555), (810, 605), (810, 755), (860, 305), (860, 505), (860, 655)]

# game_agent = GameAgent()
# game_agent.run()


# executor = InputExecutor(game_window_size=(1920, 1080))

# # executor.execute_keystroke(KeyStroke.i) # close inventory

# executor.execute_mouse_action(MouseAction.mouse_left, x=1100, y=600)
# executor.execute_mouse_action(MouseAction.mouse_right, x=200, y=200)
# executor.execute_mouse_action(MouseAction.move, x=300, y=300)

