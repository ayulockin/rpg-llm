import pytest
import weave

from rpg_agent.agents import Agent


class SimpleAgent(Agent):
    @weave.op()
    def predict(self):
        print(self.name)
        print(self.instruction)
        print(self.tools)
        print(self.chat_history)
        print(self.llm.predict(messages=[{"role": "user", "content": "Hello, how are you?"}]))
        return "I am a simple agent."


def test_simple_agent():
    agent = SimpleAgent()
    
    # Test default attributes
    assert agent.name == "base_agent"
    assert agent.model_name == "gpt-4o-mini"
    assert agent.instruction == "You are a helpful agent that can perform actions in the game."
    assert isinstance(agent.tools, list)
    assert isinstance(agent.chat_history, list)
    
    # Test prediction
    result = agent.predict()
    assert result == "I am a simple agent."
