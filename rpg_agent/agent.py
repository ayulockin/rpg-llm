import weave
import json
import cv2
import numpy as np
import random
from PIL import ImageGrab, Image

from .llm_predictor import LLMPredictor


def get_game_window():
    """Get the current game window.
    """
    return ImageGrab.grab() # defaults to whole window capture


class ScreenshotDescriptionAgent(weave.Model):
    frame_predictor: LLMPredictor
    prompt: str = """
    We are playing Divinity: Original Sin 2 game. You are provided the current frame of the game. The player or character is likely in the center of the screen. 
    
    We are trying to play this game autonomously thus we need to know what the current state of the game is. Your task is to describe the frame in great detail. Use bullet points and be very specific.
    
    - Tell the count of the objects of interest if there are multiple objects of the same type where necessary.
    - Describe the position of the objects of interest with respect to the player or character.
    - Also describe how far the object of interest is from the player or character.
    """.strip()

    @weave.op()
    def predict(self):
        response = self.frame_predictor.predict(
            system_prompt=self.prompt,
            user_prompts=[get_game_window()],
        )

        return {
            "game_frame": get_game_window(),
            "prediction": response.choices[0].message.content,
        }


class ObjectDetectionAgent(weave.Model):
    object_detector: LLMPredictor  # it can also be a standard object detection model

    prompt_task: str = """We need to detect the points of interest in the frame of the game. We are playing Divinity: Original Sin 2. Please carefully examine the frame and predict the likely positions of objects of interest in the frame.

    The frame is 1920x1080 (width, height) with the main character centered in the screen, highlighted in blue. Each object should be represented by multiple points in (x, y) pixel coordinates that indicate the approximate center or important points of the object.

    Other than the image itself, we are also providing the description of the frame to help you know the objects of interests.

    Frame description: {frame_description}
    """.strip()

    prompt_instructions: str = """
    Return ONLY a JSON object containing the object points.
        
    Follow these strict rules:
    1. Output MUST be valid JSON with no additional text
    2. Each detected object must have:
        - 'element': descriptive name of the object
        - 'points': list of (x, y) coordinates in pixel values for each point of interest within the object.
    3. Use this exact format:
        {
            "objects": [
                {
                    "element": "object_name1",
                    "points": [[x1, y1], [x2, y2], ...],
                },
                {
                    "element": "object_name2",
                    "points": [[x1, y1], [x2, y2], ...],
                }
            ]
        }
    4. Points should be precise and accurately represent the locations of each object.
    5. DO NOT include any explanation or additional text.
    """.strip()

    def parse_json(self, text: str) -> dict:
        """
        Parse JSON from text that starts with ```json and ends with ```
        """
        import re
        
        # Use regex to extract JSON content between ```json and ``` markers
        pattern = r"```json(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        
        if not match:
            return json.loads(text)
    
        # Extract and parse the JSON content
        json_str = match.group(1).strip()
        return json.loads(json_str)
    
    def draw_points(self, image, points_data, image_size=(1920, 1080)):
        # Convert the image from PIL to OpenCV format
        open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Define colors for each unique object
        color_map = {}
        # Generate random colors for variety
        def random_color():
            return (
                random.randint(0, 255),  # B
                random.randint(0, 255),  # G 
                random.randint(0, 255)   # R
            )

        # Draw points for each object
        for item in points_data:
            element = item["element"]
            points = item["points"]

            # Assign a color to each unique object type
            if element not in color_map:
                color_map[element] = random_color()

            color = color_map[element]
            
            for point in points:
                x, y = int(point[0]), int(point[1])
                
                # Draw the point on the image
                cv2.circle(open_cv_image, (x, y), 5, color, -1)
            
            # Add label with the element name and confidence near the first point of each object
            label = f"{element}"
            cv2.putText(open_cv_image, label, (int(points[0][0]), int(points[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Convert the image back from OpenCV format (BGR) to PIL format (RGB)
        result_image = Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))
        return result_image

    @weave.op()
    def predict(self, frame_description: str):
        image = get_game_window()

        system_prompt = self.prompt_task.format(frame_description=frame_description)
        system_prompt += "\n" + self.prompt_instructions

        print(system_prompt)

        response = self.object_detector.predict(
            system_prompt=system_prompt,
            user_prompts=[image],
        )

        output = {
            "game_frame": image,
            "prediction": response.choices[0].message.content,
        }

        try:
            points_data = self.parse_json(output["prediction"])
            print(points_data)
            output["points_image"] = self.draw_points(image, points_data["objects"])
        except json.JSONDecodeError:
            print("Error decoding JSON from response")
        
        return output
