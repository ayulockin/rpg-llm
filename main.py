import weave
from dotenv import load_dotenv

from rpg_agent.agent import DivinityAgent
from rpg_agent.llm_predictor import LLMClient, LLMPredictor

load_dotenv()
weave.init(project_name="ml-colabs/rpg-agent")
agent = DivinityAgent(
    frame_predictor=LLMPredictor(model_name="gpt-4o-mini", llm_client=LLMClient.OPENAI),
    game_window_size=(1920, 1080),
)
agent.predict()
