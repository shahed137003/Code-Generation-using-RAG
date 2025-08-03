
# 🧠 LangGraph-Powered Python Code Assistant

A modular, intelligent Python assistant that uses **LangGraph**, **RAG**, and **LLMs** to generate or explain code based on user prompts — deployed with an interactive web UI using **Gradio**.

---

## 🎯 Project Goal

Build a smart assistant that:
- Accepts user input in natural language.
- Understands the user's **intent** (e.g., generate or explain code).
- **Retrieves relevant examples** from datasets like [HumanEval](https://github.com/openai/human-eval) or [MBPP](https://github.com/google-research/google-research/tree/master/mbpp).
- Calls a **local or hosted LLM** to generate or explain code.
- Organizes the entire pipeline using **LangGraph’s State Machine**.

---

## 🏢 Why Use LangGraph?

Companies and developers use LangGraph for:

- ✅ **State-driven LLM orchestration**  
- ✅ **Reliable and modular** flows
- ✅ **Tool chaining** (e.g., RAG + generation + parsing)
- ✅ **Composable agents** with conditional paths
- ✅ Ideal for building **LLM-powered apps** in production

---

## 🔍 Learning Objectives for Intern Project

| Module                         | Objective                                                                 |
|-------------------------------|---------------------------------------------------------------------------|
| 📐 LangGraph Design            | Learn to build conditional state graphs and workflows                     |
| 🧠 RAG Pipeline                | Retrieve semantically similar examples to boost generation accuracy       |
| 🔄 LLM API Integration         | Format prompts and communicate with code generation models                |
| 📈 Intent Classification       | Implement smarter logic to determine user intent using ML or heuristics   |
| 🧑‍💻 Conversational Agent UX  | Wrap logic in a user-friendly Gradio web interface                        |

---

## 🎥 Helpful Resources

| Type             | Resource                                                                 |
|------------------|--------------------------------------------------------------------------|
| 📺 YouTube Talk  | [LangGraph: Build LLM Agents with State Machines (DeepLearning.AI)](https://www.youtube.com/watch?v=jGg_1h0qzaM&t=10333s) |
| 📺 Comparison    | [LangChain vs LangGraph](https://www.youtube.com/watch?v=qAF1NjEVHhY)     |
| 📖 Readable Guide| [LangGraph for Managing Complex Agents (Dev.to)](https://dev.to/jamesli/langgraph-state-machines-managing-complex-agent-task-flows-in-production-36f4) |
| ⚡ Crash Course  | [LangChain Crash Course](https://youtu.be/yF9kGESAi3M?si=mj1jIe69qjvJhH88) |
| 🧪 RAG Course    | [Building and Evaluating RAG Applications](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) |

---

## ✅ Project Tasks Breakdown

---

### Part 1: Smarter Intent Routing

**Goal**: Upgrade from basic keyword routing to an intelligent intent classification pipeline.

#### ✅ Subtasks:
- Use a **zero-shot classification model** like `facebook/bart-large-mnli`.
- Output:  
  ```json
  {
    "task": "generate_code",  // or "explain_code"
    "user_input": "<user prompt>"
  }
  ```
- Update LangGraph's `StateGraph` to branch on this classification result.

---

### Part 2: Deployment with Gradio

**Goal**: Make the assistant available via an interactive frontend.

#### ✅ Subtasks:
- Build a Gradio interface with:
  - 🧠 Textbox for user input
  - 🧾 Text output for AI-generated code or explanation
  - 🎨 Styled UI (dark/light theme, font, alignment)
- Create a Python function `chatbot(user_input)` that:
  - Uses `LangGraph` to route and process the request
  - Returns clean responses only (not full LLM dump)



---

### Part 3: RAG Evaluation on MBPP Dataset

**Goal**: Evaluate the end-to-end system (retrieval + generation) on a real Python benchmark.

#### ✅ Subtasks:
- Use [MBPP dataset](https://github.com/google-research/google-research/tree/master/mbpp) which contains:
  - `task_id`, `prompt`, `code_solution`, `test_cases`
- For each example:
  - Embed the prompt using `sentence-transformers`
  - Retrieve top-3 most similar examples from ChromaDB
  - Generate solution using the LLM
  - Execute the generated code against test cases

#### 📈 Metrics to Evaluate:
| Metric           | Description                                             |
|------------------|---------------------------------------------------------|
| ✅ Pass Rate      | Number of LLM-generated solutions passing all test cases |
| 🔁 Retrieval Quality | Whether top-3 examples are semantically close and helpful |
| 🧠 Accuracy        | Match with ground-truth solutions                        |

---

## 🔧 Tools Used

| Tool               | Purpose                             |
|--------------------|-------------------------------------|
| `LangGraph`        | State machine logic for routing     |
| `SentenceTransformers` | Semantic embeddings               |
| `ChromaDB`         | Vector database                     |
| `Transformers`     | LLMs for explanation/generation     |
| `Gradio`           | Interactive web frontend            |
| `Pandas`           | Dataset handling                    |

---

## 🧠 Models Used

| Component             | Model Name                                 |
|-----------------------|--------------------------------------------|
| Intent Classifier     | `facebook/bart-large-mnli` (HuggingFace)   |
| Code Generation LLM   | `microsoft/phi-2` (small, efficient model) |
| Embedding Model       | `all-MiniLM-L6-v2`                          |

---

## 🚀 Setup Instructions

```bash
# Install Python dependencies
pip install -U sentence-transformers chromadb pandas openai transformers langchain gradio
```


