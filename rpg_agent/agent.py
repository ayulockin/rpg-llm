import mss
import weave
from PIL import Image
from time import sleep

from rpg_agent.llm_predictor import LLMPredictor
from rpg_agent.control_interface import InputExecutor, KeyStroke
from rpg_agent.game_state.stats_catalogue import CharacterAttributes


class BaseAgent(weave.Model):
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
            img = Image.frombytes(
                "RGB", screenshot.size, screenshot.bgra, "raw", "BGRX"
            )
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


class StatsCatalogueAgent(BaseAgent):

    @weave.op()
    def crop_left_half(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        left_half_box = (0, 0, width // 2, height)
        left_half_image = image.crop(left_half_box)
        return left_half_image

    @weave.op()
    def predict(self):
        InputExecutor().execute_keystroke(KeyStroke.i)
        sleep(5)
        left_half_image = self.crop_left_half(self.get_game_window())
        return self.frame_predictor.predict(
            system_prompt="""
You are provided with a screenshot from a role playing game that shows the attributes of a character.

You are to extract the following information from the screenshot:
1. You are to extract the strength, finesse, intelligence, constitution, memory, wit as the base stats of the character.
2. You are to extract the damage max, damage min, critical chance, accuracy, dodging, physical armour current, physical armour total, magic armour current, magic armour total as the combat stats of the character.
3. You are to extract the movement, initiative, experience, next level as the action stats of the character.
4. You are to extract the fire, water, earth, air, poison as the elemental stats of the character.
5. You are to extract the vitality total, vitality current, action points, source points as the health stats of the character.
6. If a value is mentioned as a percentage, convert it to a decimal between 0 and 1.
        """,
            user_prompts=[left_half_image],
            response_model=CharacterAttributes,
        )
