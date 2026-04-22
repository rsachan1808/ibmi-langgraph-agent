# IBMi LangGraph Agent — Structured state machine with RAG

## What it does
Extends the IBMi RAG agent with explicit flow control using LangGraph.
The previous agent left all routing decisions to the LLM with a raw
while loop. This agent uses a state machine where every transition is
named, visible, and independently controllable — giving the engineer
control over what happens at each step rather than leaving it entirely
to the LLM.

## Why a state machine over a while loop

A while loop works but it is opaque. When something goes wrong you
cannot see which step failed or why. A state machine makes every
transition explicit — each node does one job, each edge is a named
decision, and every step prints what is happening.

The practical difference in debugging: when the router was broken in
development, the while loop would have returned a wrong answer silently
with no indication of the problem. The state machine printed
`[ROUTER] No tool needed → finish` when it should have printed
`[ROUTER] Tool requested → execute_tool` — revealing the exact
failure point immediately without guesswork.

This visibility is not just useful during development. In a production
IBMi environment where the agent is answering queries from developers
across the organisation, knowing exactly which node failed and why is
the difference between a 5-minute fix and a 2-hour investigation.

## Architecture

[START]
↓
[call_llm] ←─────────────────────┐
↓                              │
[router] ──→ tool_use  → [execute_tool]
↓
├──→ end_turn    → [finish]       → [END]
└──→ step_limit  → [handle_error] → [END]

### The four nodes

**call_llm** — sends the current conversation state to Claude with
available tool definitions. Serialises all response blocks to
dictionaries so they survive being stored in state across iterations.

**execute_tool** — handles ALL tool calls in a single response, not
just the first one. When Claude requests two function lookups in
parallel (e.g. comparing %DATE and %PARMS), this node executes both
and returns a matching tool_result for each. The Anthropic API requires
every tool_use block to have a corresponding tool_result — missing even
one causes a 400 error.

**finish** — extracts the final text answer from state and returns it.
Handles three content formats defensively: plain string, list of dicts
with a text block, and unexpected types — because Claude's response
format varies depending on whether tools were used.

**handle_error** — catches two failure conditions: step limit exceeded
and unhandled exceptions. Returns a structured error message rather
than crashing so the calling application can handle it gracefully.

### The router

The router is the decision point after every call_llm execution.
It makes three possible decisions:

- **tool_use** — Claude's response contains one or more tool_use
  blocks. Control passes to execute_tool. After execution, the graph
  loops back to call_llm with the tool results appended to state.

- **end_turn** — Claude's response contains only text. The agent has
  enough information to answer. Control passes to finish.

- **step_limit** — the agent has made 10 or more tool calls without
  reaching end_turn. Control passes to handle_error. Without this
  guard, a poorly worded question or ambiguous tool description could
  cause the agent to loop indefinitely — consuming API credits and
  blocking other processes in a production environment.

### State

Every node reads from and writes back to a shared state dictionary:

```python
class AgentState(TypedDict):
    question:     str   # original question, never changes
    messages:     list  # full conversation history, grows each step
    tool_result:  str   # result from most recent tool call
    final_answer: str   # populated by finish node
    error:        str   # populated by handle_error node
    steps:        int   # counts tool calls, checked by router
```

State is the memory of the graph. Because Claude has no memory between
API calls, the messages list is manually extended at each node —
appending Claude's tool requests and your tool results so Claude can
follow the reasoning thread across multiple steps.

## Key debugging lessons from building this

**Silent failures are the hardest bugs.** The most surprising problem
in development was a broken router that produced no error — it simply
routed to finish when it should have routed to execute_tool, returning
a wrong answer with no exception raised. The node trace logs revealed
it immediately. Without those logs the bug would have been invisible.

**Parallel tool calls require parallel results.** When Claude requests
two tools in a single response, you must return a tool_result for every
tool_use block before making the next API call. Returning only one
result causes a 400 error. The fix is iterating over all tool_use
blocks rather than stopping at the first.

**Serialise early.** Anthropic SDK returns typed objects (ToolUseBlock,
TextBlock) not plain dicts. LangGraph stores state between nodes and
these objects do not survive the round trip reliably. Convert everything
to plain dicts immediately in call_llm before storing in state.

## How to run it

**Install dependencies:**
```bash
pip install langchain langchain-community langchain-anthropic
pip install langchain-voyageai chromadb rank_bm25 anthropic
pip install langgraph python-dotenv
```

**Create a .env file:**
ANTHROPIC_API_KEY=your-anthropic-key
VOYAGE_API_KEY=your-voyageai-key

**Add your IBMi documentation:**
Place your IBMi reference file as `ibmi_docs.txt` in the same folder.

**Run:**
```bash
python langgraph_agent.py
```

## Example traces

**Single function lookup:**
[NODE: call_llm] Step 1 — Serialised 1 blocks: ['tool_use']
[ROUTER] Tool requested → execute_tool
[NODE: execute_tool] Found 1 tool call(s)
[NODE: call_llm] Step 2 — Serialised 1 blocks: ['text']
[ROUTER] No tool needed → finish
Answer: %DATE returns the current system date...

**Parallel function comparison:**
[NODE: call_llm] Step 1 — Serialised 3 blocks: ['text', 'tool_use', 'tool_use']
[ROUTER] Tool requested → execute_tool
[NODE: execute_tool] Found 2 tool call(s)
[NODE: call_llm] Step 2 — Serialised 1 blocks: ['text']
[ROUTER] No tool needed → finish
Answer: %DATE handles date operations, %PARMS counts parameters...

**General knowledge — no tool needed:**
[NODE: call_llm] Step 1 — Serialised 1 blocks: ['text']
[ROUTER] No tool needed → finish
Answer: The capital of Australia is Canberra...

## What comes next
Error handling and guardrails — wiring handle_error to catch tool
failures, API timeouts, and empty responses so every failure mode
has an explicit path through the graph rather than an unhandled
exception.

---
Built with: Anthropic Claude · LangGraph · LangChain · ChromaDB ·
VoyageAI Embeddings · BM25 · Python
State machine: 4 nodes · 1 conditional router · step limit guard