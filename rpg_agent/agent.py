import weave
from PIL import ImageGrab

from .llm_predictor import LLMPredictor


class ScreenshotDescriptionAgent(weave.Model):
    frame_predictor: LLMPredictor
    prompt: str = """
    We are playing Divinity: Original Sin 2 game. You are provided the current frame of the game. The character in the center of the screen is the player. 
    
    We are traying to play this game autonomously thus we need to know what the current state of the game is. Your task is to describe the frame in great detail. Use bullet points and be very specific. Tell the count of the object of interest where necessary and the position of it with respect to the player.
"""

    @weave.op()
    def get_game_window(self):
        return ImageGrab.grab() # defaults to whole window capture

    @weave.op()
    def predict(self):
        return {
            "game_frame": self.get_game_window(),
            "prediction": self.frame_predictor.predict(
                system_prompt=self.prompt,
                user_prompts=[self.get_game_window()],
            ),
        }

