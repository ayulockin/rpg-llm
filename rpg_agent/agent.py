import mss
import weave
from PIL import Image

from .llm_predictor import LLMPredictor


class ScreenshotDescriptionAgent(weave.Model):
    frame_predictor: LLMPredictor
    monitor_index: int = 2
    prompt: str = """
    We are playing Divinity: Original Sin 2 game. You are provided the current frame of the game. The character in the center of the screen is the player. 
    
    We are traying to play this game autonomously thus we need to know what the current state of the game is. Your task is to describe the frame in great detail. Use bullet points and be very specific. Tell the count of the object of interest where necessary and the position of it with respect to the player.
"""

    @weave.op()
    def get_game_window(self):
        with mss.mss() as sct:
            monitors = sct.monitors
            extended_display = monitors[self.monitor_index]
            screenshot = sct.grab(extended_display)
            img = Image.frombytes('RGB', screenshot.size, screenshot.bgra, 'raw', 'BGRX')
        return img

    @weave.op()
    def predict(self):
        return {
            "game_frame": self.get_game_window(),
            "prediction": self.frame_predictor.predict(
                system_prompt=self.prompt,
                user_prompts=[self.get_game_window()],
            ),
        }
