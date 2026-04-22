from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_classic.chains import RetrievalQA
from langchain_voyageai import VoyageAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from dotenv import load_dotenv
from pathlib import Path
import os
from typing import TypedDict

from langgraph.graph import StateGraph, END
import anthropic

# Load environment
dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

print("Loading IBMi documentation...")

loader    = TextLoader("RPG_test.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)

embeddings  = VoyageAIEmbeddings(
    voyage_api_key=os.environ.get("VOYAGE_API_KEY"),
    model="voyage-3"
)
vectorstore = Chroma.from_documents(chunks, embeddings)

bm25_retriever   = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

rag_llm  = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY")
)
qa_chain = RetrievalQA.from_chain_type(
    llm=rag_llm,
    retriever=ensemble_retriever,
    return_source_documents=True
)

print("RAG pipeline ready")

# ── State definition ──────────────────────────────────────
# This dictionary is passed between every node
# Each node reads from it and writes back to it
class AgentState(TypedDict):
    question:     str        # original question, never changes
    messages:     list       # full conversation history
    tool_result:  str        # what the last tool returned
    final_answer: str        # populated when agent is done
    error:        str        # populated if something goes wrong
    steps:        int        # counts how many tool calls happened

print("State defined")

# ── Tool definition ───────────────────────────────────────
tools = [
    {
        "name": "get_ibmi_function",
        "description": "Search IBMi RPG documentation to find built-in \
                        functions, their syntax, and usage. Use this when \
                        the user asks about IBMi functions, operators, \
                        or any RPG built-in keyword. \
                        IMPORTANT: For comparison questions involving \
                        multiple functions, call this tool separately \
                        for each function before comparing them.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The specific IBMi function name to look up, \
                                   e.g. '%DATE' or '%LEN'. Look up one function \
                                   per call."
                }
            },
            "required": ["query"]
        }
    }
]

# Utilizing RAG Pipeline
def get_ibmi_function(query):
    result = qa_chain.invoke({"query": query})
    return result["result"]

print("Tool defined")

# ── Node 1: Call the LLM ──────────────────────────────────
def call_llm(state: AgentState) -> dict:
    print(f"\n[NODE: call_llm] Step {state.get('steps', 0) + 1}")

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        tools=tools,
        messages=state["messages"]
    )

    # Serialise ALL blocks to dicts defensively
    content = []
    for block in response.content:
        # Already a dict — use as is
        if isinstance(block, dict):
            content.append(block)
        # ToolUseBlock object
        elif hasattr(block, "type") and block.type == "tool_use":
            content.append({
                "type":  "tool_use",
                "id":    block.id,
                "name":  block.name,
                "input": block.input
            })
        # TextBlock object
        elif hasattr(block, "type") and block.type == "text":
            content.append({
                "type": "text",
                "text": block.text
            })
        # Anything else — convert to string and store as text
        else:
            content.append({
                "type": "text",
                "text": str(block)
            })

    print(f"  Serialised {len(content)} blocks: {[b['type'] for b in content]}")

    return {
        "messages": state["messages"] + [
            {"role": "assistant", "content": content}
        ],
        "steps": state.get("steps", 0) + 1
    }

# ── Node 2: Execute the tool ──────────────────────────────
def execute_tool(state: AgentState) -> dict:
    print(f"[NODE: execute_tool]")

    # Get last assistant message
    last_message = state["messages"][-1]

    # Find ALL tool use blocks — not just the first one
    tool_use_blocks = [
        block for block in last_message["content"]
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]

    print(f"  Found {len(tool_use_blocks)} tool call(s)")

    # Execute every tool call and collect all results
    tool_results = []
    last_result  = ""

    for tool_use_block in tool_use_blocks:
        tool_name  = tool_use_block["name"]
        tool_input = tool_use_block["input"]
        tool_id    = tool_use_block["id"]

        print(f"  Tool: {tool_name}")
        print(f"  Input: {tool_input}")

        if tool_name == "get_ibmi_function":
            result = get_ibmi_function(tool_input["query"])
        else:
            result = f"Unknown tool: {tool_name}"

        print(f"  Result: {result}")
        last_result = result

        # Every tool_use block needs a matching tool_result
        tool_results.append({
            "type":        "tool_result",
            "tool_use_id": tool_id,
            "content":     result
        })

    # Add ALL tool results in a single user message
    updated_messages = state["messages"] + [
        {
            "role":    "user",
            "content": tool_results
        }
    ]

    return {
        "messages":    updated_messages,
        "tool_result": last_result
    }

# ── Node 3: Finish ────────────────────────────────────────
def finish(state: AgentState) -> dict:
    print(f"[NODE: finish]")

    # Extract text answer from last message
    last_message = state["messages"][-1]
    content = last_message["content"]

    # Case 1 — content is a plain string
    if isinstance(content, str):
        answer = content

    # Case 2 — content is a list of dicts
    elif isinstance(content, list):
        # Try to find a text block
        text_block = next(
            (block for block in content
             if isinstance(block, dict) and block.get("type") == "text"),
            None
        )
        if text_block:
            answer = text_block["text"]
        else:
            # No text block found — use first item as fallback
            answer = str(content[0]) if content else "No answer generated"

    # Case 3 — unexpected type, convert to string
    else:
        answer = str(content)

    print(f"  Answer: {answer[:80]}...")
    return {"final_answer": answer}

# ── Node 4: Handle errors ─────────────────────────────────
def handle_error(state: AgentState) -> dict:
    print(f"[NODE: handle_error]")
    error_msg = f"Agent stopped after {state.get('steps', 0)} steps"
    return {"final_answer": error_msg, "error": error_msg}

print("Nodes defined")

# ── Router — decides what happens after call_llm ─────────
def route_after_llm(state: AgentState) -> str:

    # Safety limit — prevent infinite loops
    if state.get("steps", 0) >= 10:
        print("[ROUTER] Step limit reached → handle_error")
        return "handle_error"

    # Get last assistant message
    last_message = state["messages"][-1]

    # Check if Claude requested a tool
    has_tool_use = any(
        block.get("type") == "tool_use"
        for block in last_message["content"]
        if isinstance(block, dict)
    )

    if has_tool_use:
        print("[ROUTER] Tool requested → execute_tool")
        return "execute_tool"
    else:
        print("[ROUTER] No tool needed → finish")
        return "finish"

print("Router defined")

# ── Build the graph ───────────────────────────────────────
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("call_llm",     call_llm)
graph.add_node("execute_tool", execute_tool)
graph.add_node("finish",       finish)
graph.add_node("handle_error", handle_error)

# Set entry point — where the graph starts
graph.set_entry_point("call_llm")

# Add conditional edge from call_llm
# router function decides which node comes next
graph.add_conditional_edges(
    "call_llm",
    route_after_llm,
    {
        "execute_tool": "execute_tool",
        "finish":       "finish",
        "handle_error": "handle_error"
    }
)

# After tool executes — always go back to call_llm
graph.add_edge("execute_tool", "call_llm")

# After finish or error — end the graph
graph.add_edge("finish",       END)
graph.add_edge("handle_error", END)

# Compile
app = graph.compile()
print("Graph compiled successfully")

# ── Run the agent ─────────────────────────────────────────
def ask_agent(question: str) -> str:
    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print(f"{'='*50}")

    initial_state = {
        "question":     question,
        "messages":     [{"role": "user", "content": question}],
        "tool_result":  "",
        "final_answer": "",
        "error":        "",
        "steps":        0
    }

    result = app.invoke(initial_state)
    return result["final_answer"]

# Test
answer = ask_agent("type your question here")
print(f"\nFinal answer: {answer}")