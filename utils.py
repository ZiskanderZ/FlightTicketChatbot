from typing import List, Union
from tqdm.auto import tqdm
import torch
from transformers import  BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.agent import AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import Tool

class Speech:

    def __init__(self, id: int, speech_line: str) -> None:

        self.id = id
        speech_types = ['customer:', 'agent:']
        for speech_type in speech_types:
            if speech_type in speech_line:
                self.speech_line = speech_line.split(speech_type)[-1].strip()
                self.speech_type = speech_type[:-1]

    def get_speech_line(self, prefix) -> str:

      if prefix:
        return self.speech_type + ':' + self.speech_line
      else:
        return self.speech_line

    def __str__(self) -> str:
        return str(self.speech_line)

    def __repr__(self) -> str:
        return str(self.speech_line)


class Conversation:

    def __init__(self, conv: list) -> None:

        self.conv = []
        for id, speech in enumerate(conv):
            speech = Speech(id, speech)
            self.conv.append(speech)

    def get_history_before_id(self, id: int, prefix=True) -> list:

        history = []
        for speech in self.conv:
            if speech.id < id:
                history.append(speech.get_speech_line(prefix))

        return history

    def get_speech_by_id(self, id: int, prefix=True) -> str:

        for speech in self.conv:
            if speech.id == id:
                return speech.get_speech_line(prefix)

        history = []
        for speech in self.conv:
            if speech.id <= id:
                history.append(speech.get_speech_line(prefix))

        return history

    def add_speech(self, speech: str) -> None:

        speech = Speech(self.get_max_id+1, speech)
        self.conv.append(speech)

    @property
    def get_max_id(self):
        max_id = -1
        for speech in self.conv:
            if speech.id > max_id:
                max_id = speech.id
        return max_id

    def __str__(self) -> str:
        return str(self.conv)

    def __repr__(self) -> str:
        return str(self.conv)


class AllConversations:

    def __init__(self, data: list) -> None:

        self.data = []
        for conv in tqdm(data):
            conv = Conversation(conv)
            self.data.append(conv)

    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self) -> str:
        return str(self.data)


class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)
    

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        answer = llm_output

        return AgentFinish(
            # Return values is generally always a dictionary with a single `output` key
            # It is not recommended to try anything else at the moment :)
            return_values={"output": answer},
            log=llm_output,
        )

def format_promt(inp: str, out: str, history: str) -> str:
  return f'''
[INST] ### Instruction: You are a helpful assistant that helps user book air ticket. Consider the chat history, which is presented below. Input it is customer message

### Chat history:
{history}

### Input:
{inp}[/INST]
{out}

'''

def create_model_and_tokenizer(model_name: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_safetensors=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

def collate(ex: str) -> [list, list]:
    inps, targets = [], []
    for i in ex:
        inp, target = i.split('[/INST]')
        inp += '[/INST]'
        inps.append(inp)
        targets.append(target)
    return inps, targets