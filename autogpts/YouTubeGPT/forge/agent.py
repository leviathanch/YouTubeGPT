import json
import pprint

from forge.sdk import (
    Agent,
    AgentDB,
    Step,
    StepRequestBody,
    Workspace,
    ForgeLogger,
    Task,
    TaskRequestBody,
    PromptEngine,
    chat_completion_request,
)

import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType, AgentExecutor, ZeroShotAgent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from langchain.chains import LLMMathChain
from langchain.llms import OpenAI
from langchain.chains import LLMChain

from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from langchain.tools import DuckDuckGoSearchRun
from langchain.memory import PostgresChatMessageHistory

from integrations.weather_tool import get_current_weather

LOG = ForgeLogger(__name__)

class ForgeAgent(Agent):

    def __init__(self, database: AgentDB, workspace: Workspace):
        """
        The database is used to store tasks, steps and artifact metadata. The workspace is used to
        store artifacts. The workspace is a directory on the file system.

        Feel free to create subclasses of the database and workspace to implement your own storage
        """
        super().__init__(database, workspace)

        load_dotenv('.env')
        self.open_ai_api = os.getenv('OPENAI_API_KEY')

        # Prompt engine allows us to use templates
        prompt_engine = PromptEngine("gpt-3.5-turbo")

        # Memory for the chat bot
        memory = ConversationBufferMemory(memory_key="chat_history", input_key='input')
        readonlymemory = ReadOnlySharedMemory(memory=memory)
        
        # Routing classification chain
        routing_kwars = { 'categories' : ['Personal', 'Casual', 'Mathematics', 'Events', 'Weather'] }
        route_prompt = PromptTemplate.from_template(prompt_engine.load_prompt("routing", **routing_kwars))
        self.route_chain = LLMChain(
            llm = OpenAI(),
            prompt = route_prompt,
            verbose = True,
            memory = readonlymemory,  # use the read-only memory to prevent the tool from modifying the memory
        )

        # Summarty chain
        summary_prompt = PromptTemplate.from_template(prompt_engine.load_prompt("summary"))
        summary_chain = LLMChain(
            llm = OpenAI(),
            prompt = summary_prompt,
            verbose = True,
            memory = readonlymemory,  # use the read-only memory to prevent the tool from modifying the memory
        )

        # The personality core
        personality_prompt = PromptTemplate(
            template = prompt_engine.load_prompt("annitta_ui"),
            input_variables=["input", "chat_history", "context"]
        )
        print(personality_prompt)
        self.personality_core = LLMChain(
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=150, # = 112 words
            ),
            prompt = personality_prompt,
            verbose = True,
            memory = memory,  # use the read-only memory to prevent the tool from modifying the memory
        )
        #self.personality_core = ChatBotWithPersonality(OpenAI(temperature=1),memory,verbose=True)

        # Then chain for doing math
        llm_math = LLMMathChain.from_llm(llm=OpenAI(temperature=0), verbose=True)

        tools = [
            Tool(
                name='Calculator',
                func=llm_math.run,
                description='useful for when you need to answer questions about math.',
                verbose=True,
            ),
            Tool(
                name = "Search",
                func = DuckDuckGoSearchRun(),
                description = "useful for when you need to answer questions about current events, data. You should ask targeted questions.",
                verbose=True,
            ),
            Tool(
               name = "Summary",
                func = summary_chain.run,
                description = "useful for when you summarize a conversation. The input to this tool should be a string, representing who will read this summary.",
                verbose=True,
            ),
            Tool(
                name = "Weather",
                func=get_current_weather,
                description="a weather tool, useful for when you're asked about the current weather. The input to this tool should be name of a given location, for which you may want to know the weather."
            ),
        ]

        # The agent for actually executing systems functions when needed
        prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
        suffix = """Begin!"

        {chat_history}
        Question: {input}
        {agent_scratchpad}"""

        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "chat_history", "agent_scratchpad"]
        )
        llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
        self.agent_chain = AgentExecutor.from_agent_and_tools(
            agent = agent,
            tools = tools,
            verbose = True,
            memory = memory
        )

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to create
        a task.

        We are hooking into function to add a custom log message. Though you can do anything you
        want here.
        """
        task = await super().create_task(task_request)

        LOG.info(
            f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
        )

        #classification = self.route_chain.run(task.input)

        #LOG.info(
        #    f"📦 Task classified as {classification}"
        #)

        return task

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        # Get the task details
        task = await self.db.get_task(task_id)
        try:
            last_steps = await self.db.get_action_history(task_id)
        except:
            last_steps = []

        classification = self.route_chain.run(step_request.input)

        LOG.info(
            f"📦 Step classified as {classification}"
        )

        # Feed the inputs to our agent
        #agent_output = self.agent_chain.run({'input': step_request.input, 'chat_history': []})

        match classification.lower().strip():
            case 'weather':
                context = "Context: "+self.agent_chain.run(input=step_request.input)
            case _:
                context = ""

        # Feed that output to ANNITTA's lovely personality:
        step_output = self.personality_core.predict(input=step_request.input, context=context)

        action = await self.db.create_action(task_id, "meow", "woof")

        step = await self.db.create_step(
            task_id = task_id,
            input = step_request,
            output = step_output,
            #is_last=True
            is_last=False
        )

        message = f'	🔄 Step executed: {step.step_id} input: {step_request}'

        if step.is_last:
            message = (
                f'	✅ Final Step completed: {step.step_id} input: {step_request}'
            )

        LOG.info(message)

        #artifact = await self.db.create_artifact(
        #    task_id = task_id,
        #    step_id = step.step_id,
        #    file_name = task_id+'out.txt',
        #    relative_path = 'artifacts/',
        #    agent_created = True,
        #)

        LOG.info(f'Received input for task {task_id}: {step_request.input}')
        LOG.info(f'Received output for task {task_id}: {step_output}')

        #self.workspace.write(task_id=task_id, path='artifacts/'+task_id+'out.txt', data=step_output.encode())

        return step

