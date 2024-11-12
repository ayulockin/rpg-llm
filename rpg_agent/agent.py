import weave
import json
import cv2
import numpy as np
from PIL import ImageGrab, Image

from .llm_predictor import LLMPredictor


def get_game_window():
    """Get the current game window.
    """
    return ImageGrab.grab() # defaults to whole window capture


class ScreenshotDescriptionAgent(weave.Model):
    frame_predictor: LLMPredictor
    prompt: str = """
    We are playing Divinity: Original Sin 2 game. You are provided the current frame of the game. The character in the center of the screen is the player. 
    
    We are trying to play this game autonomously thus we need to know what the current state of the game is. Your task is to describe the frame in great detail. Use bullet points and be very specific. Tell the count of the object of interest where necessary and the position of it with respect to the player.
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
    prompt_task: str = """We need to detect the objects of interest in the frame of the game. We are playing Divinity: Original Sin 2. Please carefully examine the frame and detect objects of interest with precise bounding boxes that closely enclose each object. 

    The frame is 1920x1080 (width, height) with the main character centered in the screen, highlighted in blue. The bounding boxes should be non-normalized (pixel values) and match the actual size and position of each detected object as accurately as possible.

    Other than the image itself, we are also providing the description of the frame to help you detect the objects better.

    Frame description: {frame_description}
    """.strip()

    prompt_instructions: str = """
    Return ONLY a JSON object containing the bounding boxes. 
        
    Follow these strict rules:
    1. Output MUST be valid JSON with no additional text
    2. Each detected object must have:
        - 'element': descriptive name of the object
        - 'bbox': [x1, y1, x2, y2] coordinates (non-normalized pixel values) in the (x_min, y_min, x_max, y_max) format.
        - 'confidence': confidence score (0-1)
    3. Use this exact format:
        {
            "objects": [
                {
                    "element": "object_name1",
                    "bbox": [x_min, y_min, x_max, y_max],
                    "confidence": 0.95
                },
                {
                    "element": "object_name2",
                    "bbox": [x_min, y_min, x_max, y_max],
                    "confidence": 0.95
                },
            ]
        }
    4. Coordinates must be precise and properly normalized
    5. DO NOT include any explanation or additional text
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
    
    
    def draw_bboxes(self, image, bboxes, image_size=(1920, 1080)):
        # Convert the image from PIL to OpenCV format
        open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Convert normalized bbox coordinates to absolute pixel values and draw bounding boxes
        width, height = image_size
        for item in bboxes:
            element = item["element"]
            bbox = item["bbox"]
            confidence = item["confidence"]

            x_min, y_min = int(bbox[0]), int(bbox[1])
            x_max, y_max = int(bbox[2]), int(bbox[3])

            # Draw rectangle and label
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(open_cv_image, (x_min, y_min), (x_max, y_max), color, 2)
            label = f"{element} ({confidence:.2f})"
            cv2.putText(open_cv_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

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
            bboxes = self.parse_json(output["prediction"])
            print(bboxes)
            output["bboxes"] = self.draw_bboxes(image, bboxes["objects"])
        except json.JSONDecodeError:
            pass

        return output
