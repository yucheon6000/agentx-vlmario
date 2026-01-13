import argparse
import asyncio
import random
import uvicorn
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    TaskState,
    UnsupportedOperationError,
)
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

# Import WFC logic
from generate_wfc import build_model, read_ascii_map, wfc_generate

class WFCMapExecutor(AgentExecutor):
    """Custom executor that generates Mario maps using Wave Function Collapse."""

    def __init__(self, reference_paths: list[Path], square_size: int = 2):
        print(f"[WFC Mode] Building model from {len(reference_paths)} reference maps...")
        ref_grids = [read_ascii_map(p) for p in reference_paths]
        self.model = build_model(ref_grids, k=square_size)
        
        # Default output size from the first reference map
        self.default_h = len(ref_grids[0])
        self.default_w = len(ref_grids[0][0])
        self.square_size = square_size

    def generate_map(self) -> str:
        """Generate a new map using WFC."""
        attempts = 20
        # WFC can sometimes fail due to contradictions; we retry with different seeds
        for i in range(attempts):
            try:
                seed = random.randint(0, 2**31 - 1)
                rng = random.Random(seed)
                lines = wfc_generate(self.model, out_w=self.default_w, out_h=self.default_h, rng=rng)
                map_content = "\n".join(lines).rstrip("\n")
                print(f"[WFC Mode] Successfully generated map on attempt {i+1} (seed: {seed})")
                return f"```ascii\n{map_content}\n```"
            except Exception as e:
                # logger.debug(f"[WFC Mode] Attempt {i+1} failed: {e}")
                continue
        
        return "```ascii\n(Failed to generate WFC map after multiple attempts)\n```"

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Execute task by generating a WFC map."""
        # Use incoming task ID if available, otherwise create new task
        msg = context.message
        if not msg:
            msg = new_agent_text_message("Map request", context_id=context.context_id)
        
        # Handle empty message text
        user_input = context.get_user_input()
        if not user_input or not user_input.strip():
            msg = new_agent_text_message("continue", context_id=msg.context_id)
        
        task_id = getattr(context, 'task_id', None)
        
        if task_id:
            updater = TaskUpdater(event_queue, task_id, context.context_id)
        else:
            task = new_task(msg)
            await event_queue.enqueue_event(task)
            updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Generating WFC map...", context_id=context.context_id),
        )
        
        # Run blocking WFC in thread to not block event loop
        map_response = await asyncio.to_thread(self.generate_map)
        
        await updater.update_status(
            TaskState.completed,
            new_agent_text_message(map_response, context_id=context.context_id),
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        raise ServerError(error=UnsupportedOperationError())

def main():
    parser = argparse.ArgumentParser(description="Run the Mario map designer WFC agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9110, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--name", type=str, default="MarioMapDesignerWFC", help="Agent display name")
    args = parser.parse_args()

    agent_card = AgentCard(
        name=args.name,
        description="Generates Mario-like ASCII levels using Wave Function Collapse (WFC).",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[],
    )

    # Hardcoded reference maps for WFC
    levels_dir = Path(__file__).parent / "levels"
    ref_paths = sorted(list(levels_dir.glob("*.txt")))
    if not ref_paths:
        print(f"Warning: No reference maps found in {levels_dir}")
        # Add at least one default if possible, or handle error
    
    executor = WFCMapExecutor(reference_paths=ref_paths)
    
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    ).build()

    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
