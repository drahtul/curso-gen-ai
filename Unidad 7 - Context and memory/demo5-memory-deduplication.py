from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from dotenv import load_dotenv
from pinecone import Pinecone
import hashlib
import os

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
USER_ID = "user-1"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pinecone_client.Index(os.getenv("PINECONE_INDEX_NAME", "semantic-memory"))
semantic_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings)

llm = ChatOpenAI(openai_api_key=openai_key, model="gpt-4.1-nano")

SIMILARITY_SCORE_THRESHOLD = 0.6


def deterministic_id(text: str, user_id: str) -> str:
    """El mismo texto normalizado + usuario siempre da el mismo id, asi que
    volver a guardar el hecho identico hace upsert en vez de duplicar."""
    normalized = text.strip().lower()
    return hashlib.sha256(f"{user_id}:{normalized}".encode()).hexdigest()


def find_closest_memory(text: str):
    """Devuelve (id, texto, score de similitud) del recuerdo mas cercano del
    usuario, o None si el indice todavia no tiene nada de este usuario."""
    vector = embeddings.embed_query(text)
    results = pinecone_index.query(
        vector=vector,
        top_k=1,
        filter={"user_id": USER_ID},
        include_metadata=True,
    )
    matches = results.get("matches", [])
    if not matches:
        return None
    match = matches[0]
    return match["id"], match["metadata"].get("text", ""), match["score"]


def classify_relation(new_fact: str, existing_fact: str) -> str:
    """Le pregunta al LLM como se relacionan dos afirmaciones sobre el mismo
    usuario. Responde exactamente una palabra: DUPLICATE, CONTRADICTION,
    UPDATE o NEW."""
    prompt = (
        "Compara dos afirmaciones sobre el mismo usuario y responde con "
        "exactamente una palabra: DUPLICATE, CONTRADICTION, UPDATE o NEW.\n\n"
        "DUPLICATE: la afirmacion nueva dice exactamente lo mismo que la "
        "existente, sin agregar ni quitar informacion (aunque este redactada "
        "distinto).\n"
        "CONTRADICTION: la afirmacion nueva contradice o invierte a la "
        "existente (deja de ser cierto lo que decia antes).\n"
        "UPDATE: la afirmacion nueva extiende, refina o amplia a la "
        "existente sin contradecirla (por ejemplo, agrega una opcion mas a "
        "una lista, o precisa un detalle). La existente queda incompleta u "
        "obsoleta frente a la nueva.\n"
        "NEW: la afirmacion nueva no tiene relacion con la existente.\n\n"
        f"Afirmacion existente: {existing_fact}\n"
        f"Afirmacion nueva: {new_fact}\n\n"
        "Responde con una sola palabra."
    )
    response = llm.invoke([SystemMessage(content=prompt)])
    answer = response.content.strip().upper()
    if "DUPLICATE" in answer:
        return "DUPLICATE"
    if "CONTRADICTION" in answer:
        return "CONTRADICTION"
    if "UPDATE" in answer:
        return "UPDATE"
    return "NEW"


def retrieve_semantic_memories(query: str, k: int = 3) -> list:
    """Busca en Pinecone los hechos/preferencias relevantes para el mensaje
    actual (usado por el nodo de carga de memoria, no por la tool de guardar)."""
    results = semantic_store.similarity_search(query, k=k, filter={"user_id": USER_ID})
    return [doc.page_content for doc in results]


@tool
def save_user_preference(preference: str) -> str:
    """
    Save a stable fact or preference about the user for future conversations
    (semantic memory, stored in Pinecone). Use this for things that stay true
    over time, e.g. "I'm vegetarian" or "I prefer short, direct answers".

    Before writing, this checks the closest existing memory and:
    - skips the write if it's an exact duplicate of something already known,
    - replaces the old memory if the new one contradicts, extends, or
      supersedes it (e.g. adding a second favorite to an existing one),
    - inserts normally if it's genuinely new information.
    """
    
    closest = find_closest_memory(preference)

    if closest is not None:
        closest_id, closest_text, score = closest
        print(f"  [dedup] recuerdo mas cercano: '{closest_text}' (score={score:.3f})")

        if score >= SIMILARITY_SCORE_THRESHOLD:
            relation = classify_relation(preference, closest_text)
            print(f"  [dedup] el LLM clasifico la relacion como: {relation}")

            if relation == "DUPLICATE":
                return f"Omitido: '{preference}' ya se conocia (duplicado de '{closest_text}')."

            if relation in ("CONTRADICTION", "UPDATE"):
                pinecone_index.delete(ids=[closest_id])
                semantic_store.add_texts(
                    texts=[preference],
                    metadatas=[{"user_id": USER_ID}],
                    ids=[deterministic_id(preference, USER_ID)],
                )
                verbo = "reemplazo (contradiccion)" if relation == "CONTRADICTION" else "actualizo (extiende el dato anterior)"
                return f"Actualizado: se {verbo} '{closest_text}' por '{preference}'."

        # NEW, o todavia no hay nada suficientemente parecido guardado
        semantic_store.add_texts(
            texts=[preference],
            metadatas=[{"user_id": USER_ID}],
            ids=[deterministic_id(preference, USER_ID)],
        )
        return f"Guardado como nuevo recuerdo: '{preference}'."


def dump_store():
    """Lista todo lo guardado del usuario. Pinecone no tiene un 'get all'
    directo como Chroma, asi que primero se listan los ids del namespace y
    luego se hace fetch de sus valores/metadata."""
    print("\n[Memoria semantica actual]")
    ids = []
    for id_batch in pinecone_index.list():
        ids.extend(id_batch)

    if not ids:
        print("  (vacia)")
        return

    fetched = pinecone_index.fetch(ids=ids)
    for id_, vector in fetched.vectors.items():
        if vector.metadata.get("user_id") == USER_ID:
            print(f"  {id_[:8]}... -> {vector.metadata.get('text')}")


tools = [save_user_preference]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools=tools)


class AgentState(TypedDict):
    """State containing the conversation plus what long-term memory found."""
    messages: Annotated[list, add_messages]
    memory_context: str


def load_memory_node(state: AgentState):
    """Read semantic memory before the agent replies."""
    last_message = state["messages"][-1]
    query = last_message.content if hasattr(last_message, "content") else str(last_message)

    semantic_results = retrieve_semantic_memories(query)

    print("\n[Memory Retrieval]")
    print(f"  Semantic (Pinecone): {semantic_results or 'none'}")

    context_parts = []
    if semantic_results:
        context_parts.append("Known facts/preferences about the user:\n" + "\n".join(f"- {s}" for s in semantic_results))

    return {"memory_context": "\n\n".join(context_parts)}


def agent_node(state: AgentState):
    """Call the LLM with the conversation plus any retrieved long-term memory."""
    memory_context = state["memory_context"]

    instructions = (
        "You have access to a long-term memory tool (save_user_preference). "
        "Call it whenever the user shares a new preference or stable fact "
        "about themselves. The tool already handles deduplication and "
        "contradictions on its own, so it needs a full, self-contained "
        "statement to work with -- never call it with a bare word or "
        "fragment (e.g. call it with 'The user's favorite programming "
        "languages are Python and C#', not with 'Python' and 'C#' as two "
        "separate calls). If the user mentions several closely related "
        "facts at once (e.g. multiple favorites of the same kind), combine "
        "them into a single call instead of one call per item."
    )
    if memory_context:
        instructions += f"\n\nHere is what you currently remember about this user:\n\n{memory_context}"

    context_messages = [SystemMessage(content=instructions)] + state["messages"]
    response = llm_with_tools.invoke(context_messages)
    return {"messages": [response]}


graph_builder = StateGraph(AgentState)

graph_builder.add_node("load_memory", load_memory_node)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "load_memory")
graph_builder.add_edge("load_memory", "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")
graph_builder.add_edge("agent", END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)


def stream_tool_responses(user_input: str, thread_id: str):
    """Stream responses. Short-term history lives in the checkpointer for
    this run only; the semantic store (Pinecone) persists across runs."""
    config = {"configurable": {"thread_id": thread_id}}

    for step in graph.stream(
        {"messages": [HumanMessage(content=user_input)], "memory_context": ""},
        config,
    ):
        print("\n--- Node Output ---")
        node_name = list(step.keys())[0]
        print(f"Node: {node_name}")
        node_state = step[node_name]

        if "messages" in node_state:
            for msg in node_state["messages"]:
                print(f"Message type: {type(msg).__name__}")
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    print(f"Tool calls: {msg.tool_calls}")
                else:
                    print(f"Content: {msg.content}")
    print()


if __name__ == "__main__":
    print("=" * 80)
    print("Agente con deduplicacion de memoria semantica")
    print("=" * 80)

    thread_id = "conversation-1"

    while True:
        user_query = input("Enter your query: ").strip()
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        if user_query.lower() == "dump":
            dump_store()
            continue
        if not user_query:
            print("Please enter a valid query.\n")
            continue

        stream_tool_responses(user_query, thread_id=thread_id)
