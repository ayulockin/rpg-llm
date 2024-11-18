import weave
from dotenv import load_dotenv
from rpg_agent.agents import Florence2ScreenshotDetectionAgent
from rpg_agent.llm_predictor import LLMPredictor


load_dotenv()
weave.init(project_name="ml-colabs/rpg-agent")
agent = Florence2ScreenshotDetectionAgent(
    model_name="microsoft/Florence-2-large",
    llm=LLMPredictor(model_name="gpt-4o"),
)
agent.predict()
