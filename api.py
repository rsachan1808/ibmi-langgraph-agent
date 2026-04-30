from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging
import time

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Lifespan — runs on startup and shutdown ───────────────
# This is where we initialise the agent once when the server
# starts — not on every request
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting IBMi RAG Agent API...")
    try:
        from langgraph_agent import ask_agent
        app.state.ask_agent = ask_agent
        logger.info("Agent loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load agent: {e}")
        raise

    yield   # server runs here

    # Shutdown
    logger.info("Shutting down IBMi RAG Agent API")

# ── Create FastAPI app ────────────────────────────────────
app = FastAPI(
    title="IBMi RPG Function Finder",
    description="AI agent that answers questions about IBMi RPG built-in functions",
    version="1.0.0",
    lifespan=lifespan
)

# ── Request and response models ───────────────────────────
class QuestionRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What does %DATE do in IBMi RPG?"
            }
        }

class AnswerResponse(BaseModel):
    question:      str
    answer:        str
    response_time: float

class HealthResponse(BaseModel):
    status:  str
    version: str

# ── Endpoints ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Check if the API is running.
    Call this first to verify the service is up.
    """
    return {
        "status":  "healthy",
        "version": "1.0.0"
    }

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Ask a question about IBMi RPG built-in functions.
    Returns a detailed answer based on the documentation.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    logger.info(f"Question received: {request.question}")

    start_time = time.time()

    try:
        answer = app.state.ask_agent(request.question)
        response_time = round(time.time() - start_time, 2)

        logger.info(
            f"Question answered in {response_time}s | "
            f"Question: {request.question[:50]}"
        )

        return {
            "question":      request.question,
            "answer":        answer,
            "response_time": response_time
        }

    except Exception as e:
        response_time = round(time.time() - start_time, 2)
        logger.error(
            f"Error answering question | "
            f"Question: {request.question} | "
            f"Error: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )