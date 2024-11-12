import weave
from PIL import ImageGrab

from .llm_predictor import LLMPredictor


class DivinityAgent(weave.Model):
    frame_predictor: LLMPredictor
    game_window_size: tuple[int, int] = (2560, 1440)

    @weave.op()
    def get_game_window(self):
        return ImageGrab.grab()

    @weave.op()
    def predict(self):
        return {
            "game_frame": self.get_game_window(),
            "prediction": self.frame_predictor.predict(
                system_prompt="You are a helpful assistant meant to describe current frame in detail.",
                user_prompts=[self.get_game_window()],
            ),
        }
