import weave
from dotenv import load_dotenv
from rpg_agent.agents import YOLOScreenshotDetectionAgent


load_dotenv()
weave.init(project_name="ml-colabs/rpg-agent")
agent = YOLOScreenshotDetectionAgent(model_name="yolo11s")
agent.predict()
