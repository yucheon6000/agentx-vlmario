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

# Import WFC logic
from generate_wfc import build_model, read_ascii_map, wfc_generate

class WFCMapExecutor(AgentExecutor):
    """Custom executor that generates Mario maps using Wave Function Collapse or loads pre-generated ones."""

    def __init__(self, reference_paths: list[Path], square_size: int = 2, load_dir: Path = None):
        print(f"[WFC Mode] Building model from {len(reference_paths)} reference maps...")
        ref_grids = [read_ascii_map(p) for p in reference_paths]
        self.model = build_model(ref_grids, k=square_size)
        
        # Default output size from the first reference map
        self.default_h = len(ref_grids[0])
        self.default_w = len(ref_grids[0][0])
        self.square_size = square_size
        self.load_dir = load_dir
        if self.load_dir:
            print(f"[WFC Mode] Pre-generated maps will be loaded from: {self.load_dir}")

    def generate_map(self, map_index: int = None) -> str:
        """Generate a new map using WFC or load from disk if map_index and load_dir are provided."""
        
        # Try loading from load_dir first if index is available
        if self.load_dir and map_index is not None:
            # Try different naming patterns: map_001.txt, map_01.txt, map_1.txt, map1.txt
            patterns = [
                f"map_{map_index:03d}.txt",
                f"map_{map_index:02d}.txt",
                f"map_{map_index}.txt",
                f"map{map_index}.txt"
            ]
            for p in patterns:
                file_path = self.load_dir / p
                # print(f"[WFC Mode] Checking for map file: {file_path}") # Debug log
                if file_path.exists():
                    try:
                        content = file_path.read_text(encoding="utf-8").strip()
                        print(f"[WFC Mode] SUCCESS: Loaded pre-generated map from {file_path}")
                        return f"```ascii\n{content}\n```"
                    except Exception as e:
                        print(f"[WFC Mode] ERROR: Failed to read {file_path}: {e}")
            print(f"[WFC Mode] FAILED: Could not find map index {map_index} in {self.load_dir} (tried patterns like {patterns[0]})")

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
        
        # Try to parse map index from user input (e.g., "map 1/25")
        map_index = None
        if user_input:
            match = re.search(r"map\s*(\d+)", user_input, re.IGNORECASE)
            if match:
                map_index = int(match.group(1))
                print(f"[WFC Mode] Detected request for map index: {map_index}")

            # Also allow dynamic folder selection via message: "folder: path/to/dir"
            folder_match = re.search(r"folder:\s*(\S+)", user_input, re.IGNORECASE)
            if folder_match:
                folder_path = folder_match.group(1)
                new_dir = Path(folder_path)
                
                # Path resolution strategy:
                # 1. Take as is (if absolute or relative to CWD)
                # 2. If not found, try relative to current script
                if not new_dir.exists():
                    alternative_dir = Path(__file__).parent / folder_path
                    if alternative_dir.exists():
                        new_dir = alternative_dir
                
                if new_dir.exists() and new_dir.is_dir():
                    self.load_dir = new_dir.resolve()
                    print(f"[WFC Mode] Load directory updated to: {self.load_dir}")
                else:
                    print(f"[WFC Mode] Warning: Requested folder does not exist: {folder_path} (resolved to {new_dir.absolute()})")

        status_msg = f"Fetching map {map_index}..." if map_index and self.load_dir else "Generating WFC map..."
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(status_msg, context_id=context.context_id),
        )
        
        # Run blocking WFC/IO in thread to not block event loop
        map_response = await asyncio.to_thread(self.generate_map, map_index=map_index)
        
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

    # Hardcoded reference maps for WFC
    levels_dir = Path(__file__).parent / "levels"
    ref_paths = sorted(list(levels_dir.glob("*.txt")))
    if not ref_paths:
        print(f"Warning: No reference maps found in {levels_dir}")
        # Add at least one default if possible, or handle error
    
    load_dir = None
    if args.load_dir:
        load_dir = Path(args.load_dir)
        # If the path provided doesn't exist relative to CWD, try relative to the script
        if not load_dir.exists():
            alternative_path = Path(__file__).parent / args.load_dir
            if alternative_path.exists():
                load_dir = alternative_path
        
        load_dir = load_dir.resolve()
        if not load_dir.is_dir():
            print(f"Warning: --load-dir '{args.load_dir}' is not a directory.")
            load_dir = None

    executor = WFCMapExecutor(reference_paths=ref_paths, load_dir=load_dir)
    
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
