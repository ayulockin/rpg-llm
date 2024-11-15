import weave
from dotenv import load_dotenv
from rpg_agent.agents import Florence2ScreenshotDetectionAgent


load_dotenv()
weave.init(project_name="ml-colabs/rpg-agent")
agent = Florence2ScreenshotDetectionAgent(model_name="microsoft/Florence-2-large")
agent.predict()
