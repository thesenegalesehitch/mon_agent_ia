import os
import sqlalchemy as sa
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import SQLChatMessageHistory
from langchain.prompts import PromptTemplate
from .tools.task_manager import init_db_tables, get_task_tool, get_tasks_tool
from .tools.external_tools import get_email_tool, get_sms_tool, get_calendar_tool
from .tools.file_manager import get_read_file_tool, get_write_file_tool
from dotenv import load_dotenv

load_dotenv()

class ApexAgent:
    def __init__(self):
        # 1. Initialisation de la BDD pour les tâches et l'historique
        db_url = (f"mysql+mysqlconnector://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@"
                  f"{os.getenv('MYSQL_HOST')}/{os.getenv('MYSQL_DB')}")
        self.engine = sa.create_engine(db_url)
        init_db_tables(self.engine)
        
        # 2. Initialisation du LLM et des embeddings
        self.llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1)
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

        # 3. Initialisation du Vector Store
        self.vector_store = self._load_or_create_vector_store()
        self.retriever = self.vector_store.as_retriever()

        # 4. Définition des outils (avec des fonctions qui les créent)
        tools = [
            get_task_tool(self.engine),
            get_tasks_tool(self.engine),
            get_email_tool(),
            get_sms_tool(),
            get_calendar_tool(),
            get_read_file_tool(),
            get_write_file_tool(),
        ]
        
        # 5. Prompt final de l'agent (version intégrée)
        prompt_template = """
        Tu es "Apex", l'architecte de ma performance cognitive. Ton rôle est de me guider de "zéro" à "héros" en optimisant ma capacité d'apprentissage. Tu peux utiliser des outils pour m'aider.
        Utilise 'get_task' pour planifier des tâches de révision.
        Utilise 'get_tasks' pour me donner un aperçu de mes tâches en cours.
        Utilise 'send_email_notification' ou 'send_sms_notification' pour m'envoyer des rappels.
        Utilise 'create_calendar_event' pour ajouter des sessions d'étude à mon agenda.
        Utilise 'read_local_file' pour consulter mes notes, et 'write_to_local_file' pour en prendre.
        
        Historique de conversation:
        {chat_history}
        
        Question de l'utilisateur: {input}
        {agent_scratchpad}
        """

        # 6. Création de l'agent
        agent_prompt = PromptTemplate.from_template(prompt_template)
        react_agent = create_react_agent(self.llm, tools, agent_prompt)
        self.agent_executor = AgentExecutor(agent=react_agent, tools=tools, verbose=True)

    def _load_or_create_vector_store(self):
        # ... (Le code de chargement du FAISS est correct et reste le même) ...
        pass

    def handle(self, user_id: str, text: str) -> str:
        # 7. Gestion de l'historique de conversation par utilisateur
        history = SQLChatMessageHistory(session_id=user_id, connection_string=f"mysql+mysqlconnector://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@{os.getenv('MYSQL_HOST')}/{os.getenv('MYSQL_DB')}")
        self.memory = ConversationBufferMemory(chat_memory=history)
        
        response = self.agent_executor.invoke({"input": text, "chat_history": self.memory.load_memory_variables({})["chat_history"]})
        self.memory.save_context({"input": text}, {"output": response['output']})
        return response['output']