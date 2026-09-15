from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from datetime import datetime, timezone
from pymongo import MongoClient
from dotenv import load_dotenv
from pinecone import Pinecone
import psycopg2
import os

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

USER_ID = "user-1"


# Semantic memory en Pinecone
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pinecone_client.Index(os.getenv("PINECONE_INDEX_NAME", "semantic-memory"))
semantic_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings, namespace="demo4-semantic-memory")


def retrieve_semantic_memories(query: str, k: int = 3) -> list:
    """Search Pinecone for facts/preferences relevant to the current message."""
    results = semantic_store.similarity_search(query, k=k, filter={"user_id": USER_ID})
    return [doc.page_content for doc in results]


# Episodic memory en MongoDB
# mongo_client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
mongo_client = MongoClient(os.getenv("MONGODB_URI", "mongodb://mongo:mongo@localhost:27017/agent_long_term_memory?authSource=admin"))
episodic_collection = mongo_client["agent_long_term_memory"]["episodic_events"]


def retrieve_episodic_memories(limit: int = 5) -> list:
    """Fetch the user's most recent events from MongoDB."""
    cursor = episodic_collection.find({"user_id": USER_ID}).sort("timestamp", -1).limit(limit)
    return [f"{doc['timestamp'].strftime('%Y-%m-%d %H:%M UTC')} - {doc['event']}" for doc in cursor]


# Procedural memory en PostgreSQL
def get_postgres_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "agent_memory"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    )


def init_procedural_memory_table():
    conn = get_postgres_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS procedural_memory (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                rule TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """
        )
    conn.commit()
    conn.close()


def retrieve_procedural_rules() -> list:
    conn = get_postgres_connection()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT rule FROM procedural_memory WHERE user_id = %s ORDER BY created_at",
            (USER_ID,),
        )
        rows = cur.fetchall()
    conn.close()
    return [row[0] for row in rows]



@tool
def save_user_preference(personal_information: str) -> str:
    """
    Save a stable personal information, fact or preference about the user for future conversations
    (semantic memory, stored in Pinecone). Use this for things that stay true
    over time, e.g. "I'm vegetarian" or "I prefer short, direct answers".
    """
    semantic_store.add_texts(texts=[personal_information], metadatas=[{"user_id": USER_ID}])
    return f"Saved preference to semantic memory: '{personal_information}'"


@tool
def save_episodic_event(event: str) -> str:
    """
    Save something that happened, mentioned by the user, tied to a specific
    moment in time (episodic memory, stored in MongoDB). Use this for things
    like "I have a job interview next Monday" or "I just adopted a dog".
    """
    episodic_collection.insert_one(
        {
            "user_id": USER_ID,
            "event": event,
            "timestamp": datetime.now(timezone.utc),
        }
    )
    return f"Saved event to episodic memory: '{event}'"


@tool
def update_agent_behavior(rule: str) -> str:
    """
    Save an instruction about HOW the agent should behave going forward
    (procedural memory, stored in PostgreSQL). Use this when the user tells
    you how they want you to act, e.g. "always answer in Spanish" or
    "keep your answers under 3 sentences".
    """
    conn = get_postgres_connection()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO procedural_memory (user_id, rule) VALUES (%s, %s)",
            (USER_ID, rule),
        )
    conn.commit()
    conn.close()
    return f"Saved behavior rule to procedural memory: '{rule}'"


tools = [save_user_preference, save_episodic_event, update_agent_behavior]
tools_by_name = {t.name: t for t in tools}
llm = ChatOpenAI(openai_api_key=openai_key, model="gpt-4.1-nano")

# El paso de extraccion de memoria usa un modelo mas capaz que el del chat:
extraction_model = ChatOpenAI(openai_api_key=openai_key, model="gpt-5.6-luna", reasoning={"effort": "none"})
extraction_llm = extraction_model.bind_tools(tools)

# La verificacion usa un modelo sin razonamiento: devuelve content como string
# plano, que es lo que espera verify_extraction().
verification_llm = ChatOpenAI(openai_api_key=openai_key, model="gpt-4.1-nano")



class AgentState(TypedDict):
    """State containing the conversation plus what long-term memory found."""
    messages: Annotated[list, add_messages]
    memory_context: str


def load_memory_node(state: AgentState):
    """Read all three long-term stores before the agent replies."""
    last_message = state["messages"][-1]
    query = last_message.content if hasattr(last_message, "content") else str(last_message)

    semantic_results = retrieve_semantic_memories(query)
    episodic_results = retrieve_episodic_memories()
    procedural_results = retrieve_procedural_rules()

    print("\n[Memory Retrieval]")
    print(f"  Semantic (Pinecone):    {semantic_results or 'none'}")
    print(f"  Episodic (MongoDB):     {episodic_results or 'none'}")
    print(f"  Procedural (Postgres):  {procedural_results or 'none'}")

    context_parts = []
    if procedural_results:
        context_parts.append("Behavior rules to follow:\n" + "\n".join(f"- {r}" for r in procedural_results))
    if semantic_results:
        context_parts.append("Known facts/preferences about the user:\n" + "\n".join(f"- {s}" for s in semantic_results))
    if episodic_results:
        context_parts.append("Recent events the user mentioned:\n" + "\n".join(f"- {e}" for e in episodic_results))

    return {"memory_context": "\n\n".join(context_parts)}


def extract_memory_node(state: AgentState):
    """Dedicated pass whose only job is deciding what to save long-term.

    Runs on every turn, separately from the conversational reply. With a
    small/cheap model like gpt-4.1-nano, relying on the same call that also
    has to produce a friendly reply is unreliable: casual mentions ("soy
    programador", "uso mucho Python") often get answered conversationally
    without ever triggering a tool call, so nothing gets saved. Splitting
    "decide what to remember" from "reply to the user" into two separate
    calls makes memory-saving consistent regardless of how chatty or subtle
    the user's phrasing is.
    """
    last_message = state["messages"][-1]
    user_text = last_message.content if hasattr(last_message, "content") else str(last_message)

    extraction_instructions = (
        "You are the memory-extraction step of a personal assistant. Your "
        "ONLY job is to look at the user's latest message and decide if it "
        "contains anything worth remembering long-term. You are not "
        "replying to the user here, so do not worry about being "
        "conversational -- just call the right tools.\n\n"
        "Call save_user_preference for stable facts/preferences about the "
        "user that they are STATING about themselves, even when mentioned "
        "casually or mixed into a greeting, e.g. 'soy programador', 'uso "
        "mucho Python', 'me llamo Danilo', 'soy vegetariano'.\n"
        "Call save_episodic_event for something tied to a specific moment "
        "in time, e.g. 'tengo una entrevista el lunes'.\n"
        "Call update_agent_behavior for instructions on how you (the "
        "assistant) should behave going forward, e.g. 'contesta siempre en "
        "espanol'.\n\n"
        "Do NOT extract anything from a question -- a message asking "
        "whether you know/remember something about the user (e.g. 'sabes a "
        "que me dedico?', 'te acuerdas de mi nombre?') contains no new "
        "fact, so do not call any tool for it, and never invent or infer a "
        "fact just to have something to save. Only extract facts the user "
        "is actually asserting about themselves.\n\n"
        "Everything you save must be a fact about the user AS A PERSON, "
        "readable on its own a year from now. NEVER save a description of "
        "the message or the conversation itself ('the user greets but does "
        "not state anything', 'the user asked about their profession', 'no "
        "especifica su profesion') -- such a description may be perfectly "
        "true about the message and is still worthless as memory.\n\n"
        "Example: the message 'hola, sabes a que me dedico?' is ONLY a "
        "question with no assertion in it -- the correct action is to call "
        "no tool at all. Do not turn the absence of information into a "
        "fact.\n\n"
        "You may call more than one tool if the message contains more than "
        "one kind of fact, or call none at all if there is nothing worth "
        "remembering (e.g. small talk, or any kind of question). When in "
        "doubt about whether something is worth remembering, do NOT save "
        "it -- a missed fact is much cheaper to fix than a wrong or "
        "hallucinated one sitting in memory.\n\n"
        f"User's latest message: {user_text}"
    )

    response = extraction_llm.invoke([SystemMessage(content=extraction_instructions)])

    print("\n[Memory Extraction]")
    if not response.tool_calls:
        print("  Nothing worth saving in this message.")
    for call in response.tool_calls:
        proposed_value = next(iter(call["args"].values()), "")
        if not verify_extraction(user_text, call["name"], proposed_value):
            print(f"  [verificacion] rechazado: {call['name']}({call['args']}) -- no esta afirmado explicitamente en el mensaje, no se guarda.")
            continue
        result = tools_by_name[call["name"]].invoke(call["args"])
        print(f"  {call['name']}({call['args']}) -> {result}")

    return {}


def verify_extraction(user_text: str, tool_name: str, proposed_value: str) -> bool:
    """Segunda pasada, deliberadamente separada de la extraccion: decide si
    lo propuesto merece entrar a la memoria permanente.

    Ojo con como se plantea la pregunta. La version original preguntaba "¿el
    mensaje afirma esto?", y dejaba pasar textos como "el usuario saluda
    pero no da informacion sobre si mismo": eso describe el mensaje con
    total precision, asi que la respuesta honesta era "si". La pregunta util
    no es si el texto es VERDADERO sobre el mensaje, sino si es un hecho
    SOBRE LA PERSONA que siga teniendo sentido leido solo dentro de un anio.
    """
    # El criterio de aceptacion cambia por tipo de memoria. Aplicar el mismo
    # a los tres es justo lo que rompio antes: pedirle a un evento episodico
    # que "siga teniendo sentido dentro de un anio" lo descalifica siempre,
    # porque un evento es temporal por definicion.
    criterion = {
        "save_user_preference": (
            "a stable fact about the user as a person -- who they are, "
            "what they like, what they are like -- one that would still be "
            "true and make sense read on its own a year from now"
        ),
        "save_episodic_event": (
            "a real event in the user's life that they reported (past, "
            "current or planned). It is expected and fine that it is tied "
            "to a moment in time and will eventually be in the past"
        ),
        "update_agent_behavior": (
            "an instruction the user gave about how the assistant should "
            "behave from now on"
        ),
    }.get(tool_name, "a fact about the user")

    prompt = (
        "You are the gate that decides what gets written into a user's "
        "permanent memory.\n\n"
        "Answer YES only if the proposed text is BOTH:\n"
        f"1. {criterion}; AND\n"
        "2. something the user actually asserted or requested in this "
        "message, not something inferred, assumed or invented.\n\n"
        "Answer NO if the proposed text:\n"
        "- describes the message or the conversation instead of the user "
        "or their life (e.g. 'the user greets but does not state "
        "anything', 'the user asked about X', 'User said buenos dias'). "
        "Such a description can be perfectly TRUE about the message and "
        "must STILL be rejected: being true is not enough.\n"
        "- records the absence of information rather than information.\n"
        "- is just a restatement or paraphrase of a question the user "
        "asked.\n"
        "- is about what you (the assistant) know, remember or can do.\n\n"
        f"User's message: {user_text}\n"
        f"Proposed to store: {proposed_value}\n\n"
        "Answer with exactly one word: YES (store it) or NO (reject it)."
    )
    response = verification_llm.invoke([SystemMessage(content=prompt)])
    return response.content.strip().upper().startswith("Y")


def agent_node(state: AgentState):
    """Call the LLM with the conversation plus any retrieved long-term memory."""
    memory_context = state["memory_context"]

    instructions = (
        "You are a friendly assistant WITH long-term memory of this user. "
        "Everything listed below is something you genuinely know about "
        "them, remembered from earlier conversations. Treat it as your own "
        "memory: answer questions about the user directly from it, and "
        "never claim you do not know something that is listed there, or "
        "that you lack memory of past conversations. Storing new memories "
        "is handled by a separate step, so you never need to do it "
        "yourself."
    )
    if memory_context:
        instructions += f"\n\nWhat you remember about this user:\n\n{memory_context}"

    context_messages = [SystemMessage(content=instructions)] + state["messages"]
    response = llm.invoke(context_messages)
    return {"messages": [response]}


graph_builder = StateGraph(AgentState)

graph_builder.add_node("load_memory", load_memory_node)
graph_builder.add_node("extract_memory", extract_memory_node)
graph_builder.add_node("agent", agent_node)

graph_builder.add_edge(START, "load_memory")
graph_builder.add_edge("load_memory", "extract_memory")
graph_builder.add_edge("extract_memory", "agent")
graph_builder.add_edge("agent", END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)


def stream_tool_responses(user_input: str, thread_id: str):
    """Stream responses. Short-term history lives in the checkpointer for
    this run only; the three long-term stores persist across runs."""
    config = {"configurable": {"thread_id": thread_id}}

    for step in graph.stream(
        {"messages": [HumanMessage(content=user_input)], "memory_context": ""},
        config,
    ):
        print("\n--- Node Output ---")
        node_name = list(step.keys())[0]
        print(f"Node: {node_name}")
        node_state = step[node_name]

        if node_state and "messages" in node_state:
            last_msg = node_state["messages"][-1]
            print(f"Message type: {type(last_msg).__name__}")
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                print(f"Tool calls: {last_msg.tool_calls}")
            else:
                print(f"Content: {last_msg.content}")
    print()


if __name__ == "__main__":
    init_procedural_memory_table()

    print("=" * 80)
    print("Agente con memoria largo plazo")
    print("=" * 80)

    thread_id = "conversation-1"

    while True:
        user_query = input("Enter your query: ").strip()
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        if not user_query:
            print("Please enter a valid query.\n")
            continue

        stream_tool_responses(user_query, thread_id=thread_id)
