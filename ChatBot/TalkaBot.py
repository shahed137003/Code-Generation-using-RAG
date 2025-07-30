import gradio as gr 
from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import pandas as pd
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import chromadb
from chromadb.config import Settings
import torch
from transformers import pipeline
from langchain_core.messages import AIMessage

import re

load_dotenv()

# ---------------------------- Agent State ----------------------------
class AgentState(TypedDict):
    chat: List[Union[HumanMessage, AIMessage]]
    chat_state: str

# -------------------------- Embeddings & DB --------------------------
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

# ----------------------------- LLM Setup -----------------------------
def model_setup():
    model_id = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map="auto" if torch.cuda.is_available() else None
    )
    return tokenizer, model

collection = load_vector_db()
tokenizer, model = model_setup()

def run_local_llm(prompt: str) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=1024
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()



def router_node(state: AgentState) -> AgentState:
    return state

def smart_routing(state: dict) -> str:
    # Initialize the zero-shot classification pipeline
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Extract the user input from the agent state
    user_input = state["chat"][-1].content  

    # Define possible task labels
    candidate_labels = ["generate_code", "explain_code"]

    # Perform zero-shot classification
    result = classifier(user_input, candidate_labels)

    # Prepare the output if needed (optional)
    output = {
        "task": result["labels"][0],
        "user_input": user_input
    }

    # Return the top predicted label (intent)
    return result["labels"][0]

def generate_code_node(state: AgentState) -> AgentState:
    import re

    user_input = state['chat'][-1].content.strip()

    # Final prompt with minimal instruction
    final_prompt = f"""### Task
Prompt:
{user_input}
Solution:
"""

    generated_output = run_local_llm(final_prompt)

    
    matches = re.findall(r"Solution:\s*\n?(def .*?)(?=\n#{2,}|$)", generated_output, re.DOTALL)

    if matches:
        final_solution = matches[-1].strip()
    else:
        final_solution = generated_output.strip()

    print("🧑‍💻 Final Code:\n", final_solution)
    state['chat'].append(AIMessage(content=final_solution))
    return state


def explain_code_node(state: AgentState) -> AgentState:
    user_code = state['chat'][-1].content.strip()

    final_prompt = f"""You are a helpful programming assistant.
Explain the following Python code clearly and concisely.

### Code
{user_code}

### Explanation:
"""

    explanation_output = run_local_llm(final_prompt)

    match = re.search(r"### Explanation:\s*\n?(.*?)(?=\n###|\Z)", explanation_output, re.DOTALL)
    if match:
        final_explanation = match.group(1).strip()
    else:
        final_explanation = explanation_output.strip()

    state['chat'].append(AIMessage(content=final_explanation))
    return state



#creating the graph 

graph = StateGraph(AgentState)  


graph.add_node("router", router_node)  
graph.add_node("generate_code", generate_code_node) 
graph.add_node("explain_code", explain_code_node) 

graph.set_entry_point("router")  




graph.add_conditional_edges(
    "router",
    smart_routing,  
    {
        "generate_code": "generate_code",
        "explain_code": "explain_code"
    }
)

graph.add_edge("generate_code", END)
graph.add_edge("explain_code", END)

compiled_graph = graph.compile()









def chatbot(user_input: str):
    # Initialize agent state
    state = {"chat": [], "chat_state": ""}
    
    # Append user input as a HumanMessage
    state["chat"].append(HumanMessage(content=user_input))
    
    # Run the graph stream
    event_stream = compiled_graph.stream(state)

    # Update state from streamed events
    for event in event_stream:
        if isinstance(event, dict) and "state" in event:
            state = event["state"]

    # Extract only the last AI response
    if state["chat"] and isinstance(state["chat"][-1], AIMessage):
        return state["chat"][-1].content.strip()
    else:
        return "🤖 No response generated."



   



demo = gr.Interface(fn=chatbot, inputs="text", outputs="text")
demo.launch()