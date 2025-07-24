# ----------------------------- Imports -----------------------------
from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ---------------------------- Agent State ----------------------------
class AgentState(TypedDict):
    chat: List[Union[HumanMessage, AIMessage]]
    chat_state: str

# ------------------------ Embeddings & DB ------------------------
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def embedding_process(text: str):
    return embedding_model.encode(text).tolist()

def load_vector_db() -> chromadb.Collection:
    df = pd.read_parquet("hf://datasets/openai/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet")
    df.drop(columns=['test', 'entry_point'], inplace=True)
    df['embedded_prompt'] = df['prompt'].map(embedding_process)

    client = chromadb.Client(Settings())
    collection = client.get_or_create_collection(name="humaneval_prompts")
    collection.add(
        documents=df['prompt'].tolist(),
        ids=df['task_id'].astype(str).tolist(),
        embeddings=df['embedded_prompt'].tolist(),
        metadatas=[{"solution": s} for s in df['canonical_solution']]
    )
    return collection

def search(collection, query_embedding):
    return collection.query(query_embeddings=[query_embedding], n_results=3)

# ----------------------------- Local LLM Setup -----------------------------
collection = load_vector_db()
model_name = "Salesforce/codet5p-770m" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda" if torch.cuda.is_available() else "cpu")

def run_local_llm(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# ----------------------------- Nodes -----------------------------
def chat_node(state: AgentState) -> AgentState:
    user_input = input("🧠 You: ")
    state['chat'].append(HumanMessage(content=user_input))
    return state

def route_decision(state: AgentState) -> str:
    last_message = state['chat'][-1].content.lower()
    if any(word in last_message for word in ["generate", "write", "create"]):
        return "generate_code"
    elif any(word in last_message for word in ["explain", "what does", "understand"]):
        return "explain_code"
    return "explain_code"

def generate_code_node(state: AgentState) -> AgentState:
    user_input = state['chat'][-1].content
    embedding = embedding_process(user_input)
    similar = search(collection, embedding)

    context = "\n".join([
        f"# Task:\n{prompt}\n# Solution:\n{meta['solution']}\n"
        for prompt, meta in zip(similar['documents'][0], similar['metadatas'][0])
    ])

    final_prompt = f"{context}\n# New Task:\n{user_input}\n# Solution:\n"
    generated_code = run_local_llm(final_prompt)
    print("\n🧑‍💻 Generated Code:\n", generated_code)
    state['chat'].append(AIMessage(content=generated_code))
    return state

def explain_code_node(state: AgentState) -> AgentState:
    user_code = state['chat'][-1].content
    embedding = embedding_process(user_code)
    results = search(collection, embedding)

    context = "\n".join([
        f"# Code:\n{code}\n# Explanation:\n{meta['solution']}\n"
        for code, meta in zip(results['documents'][0], results['metadatas'][0])
    ])

    final_prompt = f"You are a code assistant. Given examples, explain the following code.\n{context}\n# Code:\n{user_code}\n# Explanation:\n"
    explanation = run_local_llm(final_prompt)
    print("\n🧾 Explanation:\n", explanation)
    state['chat'].append(AIMessage(content=explanation))
    return state

# ---------------------------- LangGraph ----------------------------
graph = StateGraph(AgentState)
graph.add_node("chat", chat_node)
graph.add_node("router", lambda state: state)
graph.add_node("generate_code", generate_code_node)
graph.add_node("explain_code", explain_code_node)

graph.set_entry_point("chat")
graph.add_edge("chat", "router")
graph.add_conditional_edges(
    "router", route_decision,
    {
        "generate_code": "generate_code",
        "explain_code": "explain_code"
    }
)
graph.add_edge("generate_code", END)
graph.add_edge("explain_code", END)

compiled_graph = graph.compile()
initial_state = {"chat": [], "chat_state": ""}
compiled_graph.invoke(initial_state)
