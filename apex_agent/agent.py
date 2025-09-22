import os
import sqlalchemy as sa
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import SQLChatMessageHistory
from langchain.prompts import PromptTemplate
from .tools.task_manager import get_task_tool, get_tasks_tool, init_db_tables
from .tools.external_tools import get_email_tool, get_sms_tool, get_calendar_tool, get_search_tool
from .tools.file_manager import get_read_file_tool, get_write_file_tool
from dotenv import load_dotenv

load_dotenv()

class ApexAgent:
    def __init__(self):
        # 1. Initialisation de la BDD pour les tâches et l'historique
        self.db_url = (f"mysql+mysqlconnector://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@"
                  f"{os.getenv('MYSQL_HOST')}/{os.getenv('MYSQL_DB')}")
        self.engine = sa.create_engine(self.db_url)
        init_db_tables(self.engine)
        
        # 2. Initialisation du LLM et des embeddings
        self.llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.1)
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

        # 3. Initialisation du Vector Store
        self.vector_store = self._load_or_create_vector_store()
        self.retriever = self.vector_store.as_retriever()
        
        # 4. Définition des outils
        self.tools = [
            get_task_tool(self.engine),
            get_tasks_tool(self.engine),
            get_email_tool(),
            get_sms_tool(),
            get_calendar_tool(),
            get_read_file_tool(),
            get_write_file_tool(),
            get_search_tool()
        ]
        
        # Le prompt reste le même
        self.prompt_template = """
        Tu es "Apex", l'architecte de ma performance cognitive. Ton rôle est de me guider de "zéro" à "héros" en optimisant ma capacité d'apprentissage. Tu peux utiliser des outils pour m'aider.
        Utilise 'Planifier Tâche' pour planifier des tâches de révision.
        Utilise 'Afficher Tâches' pour me donner un aperçu de mes tâches en cours.
        Utilise 'Envoyer Email' ou 'Envoyer SMS' pour m'envoyer des rappels.
        Utilise 'Ajouter Événement Calendar' pour ajouter des sessions d'étude à mon agenda.
        Utilise 'Lire Fichier' pour consulter mes notes, et 'Écrire Fichier' pour en prendre.
        Utilise 'Recherche Web' pour trouver des informations sur internet.

        Historique de conversation:
        {chat_history}
        
        Question de l'utilisateur: {input}
        {agent_scratchpad}
        """

    def _load_or_create_vector_store(self):
        # Le code de chargement reste le même
        index_path = "./data/faiss_index"
        if os.path.exists(index_path):
            return FAISS.load_local(index_path, self.embeddings)
        else:
            if not os.path.exists("./data/notes.txt"):
                with open("./data/notes.txt", "w") as f:
                    f.write(" ") # Crée un fichier vide pour éviter une erreur
            with open("./data/notes.txt", "r", encoding="utf-8") as f:
                docs = f.readlines()
            if not docs:
                docs = ["Point de départ pour les notes de cours."]
            vector_store = FAISS.from_texts(docs, self.embeddings)
            vector_store.save_local(index_path)
            return vector_store

    def handle(self, user_id: str, text: str) -> str:
        # La gestion de l'historique et l'exécution de l'agent
        history = SQLChatMessageHistory(session_id=user_id, connection_string=self.db_url)
        memory = ConversationBufferMemory(chat_memory=history)
        
        agent_prompt = PromptTemplate.from_template(self.prompt_template)
        react_agent = create_react_agent(self.llm, self.tools, agent_prompt)
        agent_executor = AgentExecutor(agent=react_agent, tools=self.tools, verbose=True)

        response = agent_executor.invoke({
            "input": text,
            "chat_history": memory.load_memory_variables({})["chat_history"]
        })
        memory.save_context({"input": text}, {"output": response['output']})
        return response['output']
