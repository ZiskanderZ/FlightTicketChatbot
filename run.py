import os

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub
from config import hf_token_read
from utils import CustomPromptTemplate, CustomOutputParser

os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token_read

model = HuggingFaceHub(
    repo_id='zkv/llama-2-7b-chat-hf-air-assistant', model_kwargs={"temperature": 0.5, "max_length": 64}
)

tools = [
    Tool(
        name='AirSearch',
        func=lambda x: 'Airbus #7053, S7 #8095, NorthWind # 4563',  # Mock function
        description="useful for when you need flights for the user after asked him"
    )
]

# Set up the base template
template_with_history = """You are a helpful assistant that helps user book air ticket. You must ask the human the following info and remember: [departure city],  [arrival city], [date], [time].

You have access to the following tools:

{tools}

Use the following format:

Begin! Remember to ask questions and dont generate User answer. When user select suitable flights you must say goodbye to him.

Previous conversation history:
{history}

New query from human: {input}
{agent_scratchpad}"""

prompt_with_history = CustomPromptTemplate(
    template=template_with_history,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps", "history"]
)

output_parser = CustomOutputParser()

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=model, prompt=prompt_with_history)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["Observation:"],
    allowed_tools=tool_names
)
memory=ConversationBufferWindowMemory(k=10)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)

for _ in range(10):
    q = input()
    print(agent_executor.run(q))


