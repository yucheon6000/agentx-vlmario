import argparse
import asyncio
import random
import re
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

# from generate_wfc import build_model, read_ascii_map, wfc_generate

class WFCMapExecutor(AgentExecutor):
    """Custom executor that loads pre-generated Mario maps sequentially."""

    def __init__(self, load_dir: Path):
        self.load_dir = load_dir
        self.current_index = 1
        
        if self.load_dir:
            print(f"[Map Loader] Maps will be loaded from: {self.load_dir}")
        else:
            print("[Map Loader] Warning: No load directory provided.")

    def load_next_map(self) -> str:
        """Load the next map from disk and increment index."""
        map_index = self.current_index
        
        if not self.load_dir:
            return "```ascii\n(Error: No load directory configured)\n```"

        # Try different naming patterns: map_001.txt, map_01.txt, map_1.txt, map1.txt
        patterns = [
            f"map_{map_index:03d}.txt",
            f"map_{map_index:02d}.txt",
            f"map_{map_index}.txt",
            f"map{map_index}.txt"
        ]
        
        for p in patterns:
            file_path = self.load_dir / p
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding="utf-8").strip()
                    print(f"[Map Loader] SUCCESS: Loaded map {map_index} from {file_path}")
                    self.current_index += 1
                    if self.current_index > 25:
                        print("[Map Loader] Reached map 25, resetting counter to 1.")
                        self.current_index = 1
                    return f"```ascii\n{content}\n```"
                except Exception as e:
                    print(f"[Map Loader] ERROR: Failed to read {file_path}: {e}")
        
        err_msg = f"(Error: Could not find map file for index {map_index} in {self.load_dir})"
        print(f"[Map Loader] FAILED: {err_msg}")
        return f"```ascii\n{err_msg}\n```"

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
        
        # Prepare updater
        task_id = getattr(context, 'task_id', None)
        if task_id:
            updater = TaskUpdater(event_queue, task_id, context.context_id)
        else:
            task = new_task(msg)
            await event_queue.enqueue_event(task)
            updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        print(f"[Map Loader] Handling request. Current counter: {self.current_index}")
        
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Loading map {self.current_index}...", context_id=context.context_id),
        )
        
        # Load map from thread
        map_response = await asyncio.to_thread(self.load_next_map)
        
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
    parser.add_argument("--load-dir", type=str, help="Directory to load pre-generated maps from")
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

    # Determine reference and load directory
    # Priority: 1. --load-dir argument, 2. Default 'wfc_sq2' folder
    initial_dir = args.load_dir if args.load_dir else "wfc_sq2"
    load_dir = Path(initial_dir)
    
    if not load_dir.is_absolute():
        # Try relative to CWD first, then relative to script
        if not load_dir.exists():
            alternative_path = Path(__file__).parent / initial_dir
            if alternative_path.exists():
                load_dir = alternative_path
    
    load_dir = load_dir.resolve()
    if not load_dir.exists() or not load_dir.is_dir():
        print(f"Error: Directory '{initial_dir}' not found. Cannot load maps.")
        exit(1)
    else:
        print(f"[Map Loader] Initialized with directory: {load_dir}")

    executor = WFCMapExecutor(load_dir=load_dir)
    
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
