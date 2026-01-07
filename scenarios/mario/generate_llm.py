#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate mutated Mario-style ASCII maps and save them into a folder named by the model.
This script references `ASCII_GUIDELINES` and available manual maps in the same directory.

Usage examples:
  python generate_maps.py --model gpt-5.2 --count 25
  python generate_maps.py --model mymodel --count 25 --seed 1234

The saved files are named map_001.txt ... map_025.txt and each file contains the map wrapped
in the same fenced-code style the agent uses (```ascii ... ```).
"""

from __future__ import annotations

import argparse
import os
import random
import re
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Import prompt and level information from the existing module
try:
    from map_designer import ASCII_GUIDELINES, LEVELS_DIR, MANUAL_MAPS
except Exception:
    # If direct import fails (when running from a different cwd), try to locate the file
    import importlib.util
    import sys
    md_path = Path(__file__).resolve().parent / "map_designer.py"
    spec = importlib.util.spec_from_file_location("map_designer", str(md_path))
    map_designer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(map_designer)  # type: ignore
    ASCII_GUIDELINES = map_designer.ASCII_GUIDELINES
    LEVELS_DIR = map_designer.LEVELS_DIR
    MANUAL_MAPS = map_designer.MANUAL_MAPS


load_dotenv()

# NOTE: use \s to match whitespace after the opening fence
CODE_BLOCK_RE = re.compile(r"```(?:ascii|text)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
MAP_ASCII_GUIDE_PATH = PROMPTS_DIR / "map_ascii_guide.md"


def _read_text_if_exists(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


MAP_ASCII_GUIDE = _read_text_if_exists(MAP_ASCII_GUIDE_PATH)


def build_llm_map_prompt(map_number: int, total_maps: int) -> str:
    """Build the map-generation prompt to send to the LLM."""
    # Include ASCII_GUIDELINES at the top like a system instruction.
    return f"""{ASCII_GUIDELINES.strip()}

# Map Generation Request

Task: Generate map {map_number}/{total_maps}

## Instructions
- Return ONLY the ASCII map in a fenced code block: ```ascii ... ```
- Include 'M' (start) and 'F' (exit flag)
- Make each row 70-90 characters wide
- All rows must be exactly the same length (pad with '-')
- Make the map playable within 60 seconds

## ASCII Tile Reference
{MAP_ASCII_GUIDE}

## Response Format
```ascii
[Your map here]
```
"""


def extract_and_normalize_ascii_map(response_text: str, pad_char: str = "-") -> List[str]:
    match = CODE_BLOCK_RE.search(response_text or "")
    if not match:
        raise ValueError("Map response is missing an ASCII code block.")

    content = match.group(1).strip("\n")
    # Drop blank lines to make width calculation stable.
    lines = [line.rstrip("\r\n") for line in content.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Empty map submitted.")

    width = max(len(line) for line in lines)
    normalized = [line.ljust(width, pad_char) for line in lines]
    return normalized


def load_source_maps() -> List[List[str]]:
    """Load the manual map files as lists of lines.
    Returns a list of maps, where each map is a list of strings (lines).
    """
    maps = []
    for fname in MANUAL_MAPS:
        p = LEVELS_DIR / fname
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8").rstrip("\n")
        lines = text.splitlines()
        maps.append(lines)
    return maps


def collect_tiles(maps: List[List[str]]) -> List[str]:
    tiles = set()
    for m in maps:
        for line in m:
            tiles.update(list(line))
    tiles.discard("\n")
    return sorted(tiles)


def roll_line(s: str, shift: int) -> str:
    if not s:
        return s
    shift = shift % len(s)
    return s[-shift:] + s[:-shift] if shift else s


def mutate_map(lines: List[str], tiles: List[str], replace_prob: float = 0.02, flip_prob: float = 0.05) -> List[str]:
    """Return a mutated copy of the provided map lines.

    - Small random replacements with probability replace_prob per tile
    - Optional horizontal flip with probability flip_prob
    - Small horizontal roll shift
    """
    rnd = random.random()
    mutated = [list(line) for line in lines]

    # Small horizontal roll
    max_shift = max(1, max((len(l) for l in lines), default=1) // 10)
    shift = random.randint(-max_shift, max_shift)
    if shift:
        mutated = [list(roll_line(''.join(l), shift)) for l in mutated]

    # Flip horizontally sometimes
    if random.random() < flip_prob:
        mutated = [list(reversed(l)) for l in mutated]

    # Tile replacement
    candidates = [t for t in tiles if t != '\n']
    for y, row in enumerate(mutated):
        for x, ch in enumerate(row):
            if random.random() < replace_prob:
                # prefer background '-' less frequently replaced by background
                new = random.choice(candidates)
                row[x] = new

    # Add small blocks/platforms occasionally
    if random.random() < 0.2:
        h = len(mutated)
        w = len(mutated[0]) if h else 0
        if h and w:
            for _ in range(random.randint(1, 4)):
                by = random.randint(0, h - 1)
                bx = random.randint(0, max(0, w - 3))
                block_char = random.choice(["X", "x", "Q", "E", "S"]) if any(c in tiles for c in ["X", "x", "Q", "E", "S"]) else random.choice(candidates)
                blen = random.randint(1, min(6, w - bx))
                for i in range(blen):
                    mutated[by][bx + i] = block_char

    return [''.join(row) for row in mutated]


def save_map_text(out_dir: Path, idx: int, map_lines: List[str]):
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"map_{idx:03d}.txt"
    p = out_dir / name
    # The agent wraps maps in a fenced ascii block; replicate that format
    content = "```ascii\n" + "\n".join(map_lines).rstrip('\n') + "\n```\n"
    p.write_text(content, encoding="utf-8")
    return p


def generate_map_via_google_genai(prompt: str, model: str, temperature: float = 0.7) -> str:
    """Call the google-genai SDK directly and return the response text."""
    try:
        from google import genai
        from google.genai import types as genai_types
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Couldn't import google-genai. Verify dependencies are installed (e.g., `uv sync`)."
        ) from e

    client = genai.Client()
    resp = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=genai_types.GenerateContentConfig(
            temperature=temperature,
        ),
    )

    # In most cases, the aggregated response text is available on `.text`.
    text = getattr(resp, "text", None)
    if not text:
        # Fallback: at least return something debuggable.
        raise RuntimeError("LLM response is missing `.text`")
    return text


def list_google_genai_models() -> List[str]:
    """Return a list of available Google GenAI model names for current credentials."""
    try:
        from google import genai
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Couldn't import google-genai. Verify dependencies are installed (e.g., `uv sync`)."
        ) from e

    client = genai.Client()
    names: List[str] = []
    for m in client.models.list():
        name = getattr(m, "name", None)
        if name:
            names.append(str(name))
    return names


def _default_api_model_google() -> str:
    """Default model name for Google GenAI calls."""
    return os.getenv("MARIO_API_MODEL") or os.getenv("GOOGLE_GENAI_MODEL") or "gemini-1.5-flash"


def generate_map_via_openai(prompt: str, model: str, temperature: float = 0.7) -> str:
    """Call OpenAI Chat Completions directly and return the response text.

    NOTE: This repo may or may not include the `openai` package.
    If it's missing, we raise a clear error.
    """
    try:
        from openai import OpenAI
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Couldn't import `openai`. To use OpenAI, install it (e.g., `uv add openai` or pip)."
        ) from e

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            # Strong system instruction plus a user request.
            {"role": "system", "content": "You generate Mario ASCII levels. Respond with ONLY one ```ascii``` fenced code block."},
            {"role": "user", "content": prompt},
        ],
    )

    # SDK response structure: choices[0].message.content
    content = None
    try:
        content = resp.choices[0].message.content
    except Exception:
        content = None

    if not content:
        raise RuntimeError("OpenAI response is missing choices[0].message.content")

    return content


def main():
    parser = argparse.ArgumentParser(
        description="Generate ASCII Mario maps (LLM API or mutation) and save into a model-named folder."
    )
    parser.add_argument(
        "--provider",
        choices=["google", "openai"],
        default=os.getenv("MARIO_PROVIDER", "google"),
        help="LLM provider to call directly (default: env MARIO_PROVIDER or 'google').",
    )
    parser.add_argument(
        "--api-model",
        default=None,
        help="Actual provider model name. If omitted, uses provider-specific default/env.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="(google only) List available google-genai models for your credentials and exit.",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=25,
        help="Number of maps to generate (default: 25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible output",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Optional root directory to place the model folder. By default uses the levels parent directory.",
    )

    parser.add_argument(
        "--mode",
        choices=["llm", "mutate"],
        default="llm",
        help="Generation mode: 'llm' calls the provider API; 'mutate' uses local random mutations.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="LLM sampling temperature (llm mode only)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per map if response is invalid (llm mode only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call external APIs; instead write a simple deterministic placeholder map.",
    )

    args = parser.parse_args()

    if args.list_models:
        if args.provider != "google":
            raise SystemExit("--list-models is only supported for the google provider.")
        if not os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").upper() != "TRUE":
            print("Warning: GOOGLE_API_KEY is not set. If you are not using Vertex AI, set it in .env")
        models = list_google_genai_models()
        print("Available models:")
        for n in models:
            print("-", n)
        return

    # Provider-specific default API model
    if args.provider == "google":
        api_model = args.api_model or _default_api_model_google()
    else:
        api_model = args.api_model or os.getenv("OPENAI_MODEL") or os.getenv("MARIO_API_MODEL") or "gpt-4o-mini"

    if args.seed is not None:
        random.seed(args.seed)

    base = Path(args.out_root) if args.out_root else LEVELS_DIR.parent
    out_dir = base / args.api_model
    print(f"Saving {args.count} maps into: {out_dir}")

    if args.mode == "llm":
        print(f"Provider: {args.provider}")
        print(f"Using API model: {api_model}")

    if args.mode == "mutate":
        src_maps = load_source_maps()
        if not src_maps:
            print("No source maps found in", LEVELS_DIR)
            return
        tiles = collect_tiles(src_maps)
        print(f"Loaded {len(src_maps)} source maps, found {len(tiles)} tile characters.")

        for i in range(1, args.count + 1):
            src = random.choice(src_maps)
            mutated = mutate_map(src, tiles)
            p = save_map_text(out_dir, i, mutated)
            print(f"Wrote {p}")
        print("Done.")
        return

    # LLM mode
    if not args.dry_run:
        if args.provider == "google":
            if not os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").upper() != "TRUE":
                print("Warning: GOOGLE_API_KEY is not set. If you are not using Vertex AI, set it in .env")
        else:
            if not os.getenv("OPENAI_API_KEY"):
                print("Warning: OPENAI_API_KEY is not set. If you are using OpenAI, set it in .env")

    total = args.count
    for i in range(1, total + 1):
        prompt = build_llm_map_prompt(i, total)

        last_err: Optional[Exception] = None
        for attempt in range(1, args.max_retries + 1):
            try:
                if args.dry_run:
                    fake = "```ascii\n" + ("M" + ("-" * 68) + "F") + "\n" + ("X" * 70) + "\n```"
                    map_lines = extract_and_normalize_ascii_map(fake, pad_char="-")
                else:
                    if args.provider == "google":
                        text = generate_map_via_google_genai(
                            prompt=prompt,
                            model=api_model,
                            temperature=args.temperature,
                        )
                    else:
                        text = generate_map_via_openai(
                            prompt=prompt,
                            model=api_model,
                            temperature=args.temperature,
                        )
                    map_lines = extract_and_normalize_ascii_map(text, pad_char="-")

                p = save_map_text(out_dir, i, map_lines)
                print(f"Wrote {p}")
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"Map {i:03d}: attempt {attempt}/{args.max_retries} failed: {e}")

        if last_err is not None:
            raise SystemExit(f"Failed to generate map {i:03d} after {args.max_retries} attempts: {last_err}")

    print("Done.")


if __name__ == "__main__":
    main()
