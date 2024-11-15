import weave
from dotenv import load_dotenv
from rpg_agent.agents import ScreenshotDetectionAgent


load_dotenv()
weave.init(project_name="ml-colabs/rpg-agent")
agent = ScreenshotDetectionAgent()
agent.predict(prompts=[["wooden stairs"], ["human"]])
