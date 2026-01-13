import argparse
import uvicorn
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from a2a.types import (
    AgentCapabilities,
    AgentCard,
)

SYSTEM_INSTRUCTION = """
You are a professional Mario map designer. 
Your goal is to design high-quality, playable, and aesthetically pleasing Mario-style ASCII platformer maps.

## Constraints & Requirements:
1. Return ONLY the ASCII map wrapped in a fenced code block: ```ascii ... ```
2. Include 'M' (Mario start position) and 'F' (Exit Flag position).
3. All rows must be EXACTLY the same length (pad with '-' for empty space).
4. Ensure the level is playable and has a logical flow from start to finish.
5. Use the ASCII tile reference provided in the request messages to compose your level.
"""

def main():
    parser = argparse.ArgumentParser(description="Run the Mario map designer LLM agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9110, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--name", type=str, default="MarioMapDesignerLLM", help="Agent name")
    args = parser.parse_args()

    model_name = os.getenv("MODEL", "gemini-2.0-flash")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("mario_map_designer")

    logger.info(f"Starting Mario Map Designer ({args.name})")
    logger.info(f"Using model: {model_name}")
    logger.info(f"Endpoint: http://{args.host}:{args.port}")

    root_agent = Agent(
        name=args.name,
        model=model_name,
        description="Generates Mario-like ASCII levels using an LLM.",
        instruction=SYSTEM_INSTRUCTION.strip(),
    )

    agent_card = AgentCard(
        name=args.name,
        description="Generates Mario-like ASCII levels within a fenced code block.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[],
    )

    a2a_app = to_a2a(root_agent, agent_card=agent_card)
    uvicorn.run(a2a_app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
