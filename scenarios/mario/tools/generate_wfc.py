#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wave Function Collapse (WFC) based ASCII Mario map generator.

Contract
- Input: reference ASCII map text file(s) (e.g. scenarios/mario/levels/text_level_0.txt)
- Output: N generated maps saved as `map_###.txt` under an output folder
- Core idea: learn local NxN patterns from reference, then synthesize a new grid

Notes
- This is an "overlapping" WFC variant for character grids.
- It enforces local adjacency constraints between extracted patterns.
- It does not guarantee playability; it only matches local style/grammar.

Usage examples:
  uv run python scenarios/mario/generate_wfc.py \
    --reference scenarios/mario/levels/text_level_0.txt \
    --out-dir scenarios/mario/wfc_out \
    --count 25

  # same size as reference, with deterministic seed
  uv run python scenarios/mario/generate_wfc.py \
    --reference scenarios/mario/levels/text_level_0.txt \
    --same-size \
    --seed 123 \
    --count 5

"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple


Coord = Tuple[int, int]
Pattern = Tuple[str, ...]  # tuple of rows


def read_ascii_map(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8").rstrip("\n")
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"Empty map: {path}")
    width = max(len(l) for l in lines)
    # normalize width (reference maps should already be equal-width)
    return [l.ljust(width, "-") for l in lines]


def write_ascii_map(path: Path, lines: Sequence[str], fenced: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines).rstrip("\n") + "\n"
    if fenced:
        content = f"```ascii\n{content}```\n"
    path.write_text(content, encoding="utf-8")


def iter_patterns(grid: Sequence[str], n: int) -> Iterable[Pattern]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    if h < n or w < n:
        raise ValueError(f"Reference too small for n={n}: {h}x{w}")
    for y in range(0, h - n + 1):
        for x in range(0, w - n + 1):
            yield tuple(row[x : x + n] for row in grid[y : y + n])


def pattern_edges(p: Pattern) -> Tuple[str, str, str, str]:
    # top, bottom, left, right
    top = p[0]
    bottom = p[-1]
    left = "".join(row[0] for row in p)
    right = "".join(row[-1] for row in p)
    return top, bottom, left, right


@dataclass(frozen=True)
class WFCModel:
    # square constraint size (k x k)
    k: int
    # Allowed kxk blocks learned from references (NO rotations/reflections)
    allowed_blocks: FrozenSet[Tuple[str, ...]]
    # Frequency of each block in reference (pattern weights)
    block_weights: Dict[Tuple[str, ...], float]
    # Allowed 1D rows (length k) and cols (length k) extracted from those blocks
    allowed_rows: FrozenSet[str]
    allowed_cols: FrozenSet[str]
    # Per-value weights for choosing edge labels (based on frequency)
    row_weights: Dict[str, float]
    col_weights: Dict[str, float]
    # Per-character weights learned from reference tile frequency
    char_weights: Dict[str, float]


def build_model(reference_grids: List[List[str]], k: int) -> WFCModel:

    if k < 2:
        raise ValueError("square size k must be >= 2")

    blocks_count: Dict[Tuple[str, ...], int] = defaultdict(int)
    for grid in reference_grids:
        h = len(grid)
        w = len(grid[0]) if h else 0
        if h < k or w < k:
            raise ValueError(f"Reference too small for k={k}: {h}x{w}")
        for y in range(0, h - k + 1):
            for x in range(0, w - k + 1):
                block = tuple(row[x : x + k] for row in grid[y : y + k])
                blocks_count[block] += 1

    if not blocks_count:
        raise ValueError("No blocks learned from references")

    allowed_blocks = frozenset(blocks_count.keys())
    block_weights: Dict[Tuple[str, ...], float] = {b: float(c) for b, c in blocks_count.items()}

    row_counts: Dict[str, int] = defaultdict(int)
    col_counts: Dict[str, int] = defaultdict(int)

    # rows/cols of length k that appear inside allowed blocks
    for block, c in blocks_count.items():
        for r in block:
            row_counts[r] += c
        for xi in range(k):
            col = "".join(block[yi][xi] for yi in range(k))
            col_counts[col] += c

    allowed_rows = frozenset(row_counts.keys())
    allowed_cols = frozenset(col_counts.keys())

    row_weights = {r: float(cnt) for r, cnt in row_counts.items()}
    col_weights = {c: float(cnt) for c, cnt in col_counts.items()}

    # --- character frequency over the full reference grids ---
    char_counts: Dict[str, int] = defaultdict(int)
    for grid in reference_grids:
        for row in grid:
            for ch in row:
                char_counts[ch] += 1

    # ensure all chars that appear in blocks have a weight, even if counts somehow miss
    char_weights: Dict[str, float] = {ch: float(cnt) for ch, cnt in char_counts.items()}

    return WFCModel(
        k=k,
        allowed_blocks=allowed_blocks,
        block_weights=block_weights,
        allowed_rows=allowed_rows,
        allowed_cols=allowed_cols,
        row_weights=row_weights,
        col_weights=col_weights,
        char_weights=char_weights,
    )


def weighted_choice(rng: random.Random, items: Sequence[int], weights: Dict[int, float]) -> int:
    total = 0.0
    for i in items:
        total += float(weights.get(i, 1.0))
    r = rng.random() * total
    acc = 0.0
    for i in items:
        acc += float(weights.get(i, 1.0))
        if acc >= r:
            return i
    return items[-1]


def weighted_choice_str(rng: random.Random, items: Sequence[str], weights: Dict[str, float]) -> str:
    total = 0.0
    for it in items:
        total += float(weights.get(it, 1.0))
    r = rng.random() * total
    acc = 0.0
    for it in items:
        acc += float(weights.get(it, 1.0))
        if acc >= r:
            return it
    return items[-1]


def neighbors(pos: Coord, w: int, h: int) -> Iterable[Tuple[str, Coord]]:
    x, y = pos
    if y > 0:
        yield "N", (x, y - 1)
    if y + 1 < h:
        yield "S", (x, y + 1)
    if x > 0:
        yield "W", (x - 1, y)
    if x + 1 < w:
        yield "E", (x + 1, y)


def shannon_entropy(options: Set[int], weights: Dict[int, float]) -> float:
    # classic WFC entropy: log(sum w) - sum(w log w)/sum(w)
    if not options:
        return float("inf")
    ws = [float(weights.get(i, 1.0)) for i in options]
    sw = sum(ws)
    if sw <= 0:
        return float("inf")
    return math.log(sw) - sum(w * math.log(w) for w in ws) / sw


def shannon_entropy_labels(options: Set[str], weights: Dict[str, float]) -> float:
    if not options:
        return float("inf")
    ws = [float(weights.get(v, 1.0)) for v in options]
    sw = sum(ws)
    if sw <= 0:
        return float("inf")
    return math.log(sw) - sum(w * math.log(w) for w in ws) / sw


class Contradiction(Exception):
    pass


def wfc_generate(
    model: WFCModel,
    out_w: int,
    out_h: int,
    rng: random.Random,
    max_steps: int = 200_000,
) -> List[str]:
    """Generate an output grid of size out_h x out_w (characters) using k×k square constraints.

    Strategy
    - Represent unknowns as labels on grid edges:
      - H edges: between (x,y) and (x+1,y) => size out_h x (out_w-1)
      - V edges: between (x,y) and (x,y+1) => size (out_h-1) x out_w
    - A k×k block implies constraints on its internal rows/cols of length k.
    - We enforce: for every top-left (i,j), the k horizontal row-strings and k vertical col-strings
      must correspond to SOME allowed k×k block.

    Note: No rotations/reflections are ever added to allowed blocks.
    """
    k = model.k
    if out_w < k or out_h < k:
        raise ValueError(f"Output too small for k={k}: {out_h}x{out_w}")

    # Edge variables domains
    # H[y][x] is the character at (x+1,y) given (x,y): actually edge label itself is a character.
    # But for k×k constraints, we need rows/cols strings length k; easier:
    # define cell chars directly as variables with domains from reference alphabet.

    # Derive alphabet from allowed blocks
    alphabet: Set[str] = set()
    for block in model.allowed_blocks:
        for row in block:
            alphabet.update(row)
    if not alphabet:
        raise ValueError("Empty alphabet")

    # Local weights restricted to alphabet (fallback weight=1.0)
    global_char_weights: Dict[str, float] = {ch: float(model.char_weights.get(ch, 1.0)) for ch in alphabet}

    # Every cell starts with all possible characters
    wave: List[List[Set[str]]] = [[set(alphabet) for _ in range(out_w)] for _ in range(out_h)]
    collapsed: List[List[Optional[str]]] = [[None for _ in range(out_w)] for _ in range(out_h)]

    def possible_blocks_for(tlsx: int, tlsy: int) -> List[Tuple[str, ...]]:
        """Return allowed blocks compatible with current cell domains for block at (tlsx,tlsy)."""
        out: List[Tuple[str, ...]] = []
        for block in model.allowed_blocks:
            ok = True
            for dy in range(k):
                for dx in range(k):
                    chs = wave[tlsy + dy][tlsx + dx]
                    if block[dy][dx] not in chs:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                out.append(block)
        return out

    def local_char_distribution_for_cell(cx: int, cy: int) -> Dict[str, float]:
        weights: Dict[str, float] = defaultdict(float)

        min_x = max(0, cx - (k - 1))
        max_x = min(out_w - k, cx)
        min_y = max(0, cy - (k - 1))
        max_y = min(out_h - k, cy)

        for ty in range(min_y, max_y + 1):
            for tx in range(min_x, max_x + 1):
                candidates = possible_blocks_for(tx, ty)
                if not candidates:
                    continue
                # index inside the block for this cell
                dx = cx - tx
                dy = cy - ty
                for b in candidates:
                    ch = b[dy][dx]
                    # use pattern frequency as weight
                    weights[ch] += float(model.block_weights.get(b, 1.0))

        # Keep only currently allowed options
        allowed = wave[cy][cx]
        if allowed:
            weights = {ch: w for ch, w in weights.items() if ch in allowed and w > 0}

        # fallback: if no block-derived info (should be rare), use global char frequency
        if not weights:
            return {ch: float(global_char_weights.get(ch, 1.0)) for ch in allowed} if allowed else {}

        # light smoothing with global character frequency so rare-but-valid chars don't become impossible
        smoothed: Dict[str, float] = {}
        for ch in allowed:
            smoothed[ch] = float(weights.get(ch, 0.0)) + 0.01 * float(global_char_weights.get(ch, 1.0))
        return smoothed

    def propagate(queue: List[Coord]) -> None:
        steps = 0
        while queue:
            steps += 1
            if steps > max_steps:
                raise RuntimeError("Propagation exceeded max_steps; likely stuck")

            cx, cy = queue.pop()
            # Any k×k block that includes (cx,cy) might be affected.
            min_x = max(0, cx - (k - 1))
            max_x = min(out_w - k, cx)
            min_y = max(0, cy - (k - 1))
            max_y = min(out_h - k, cy)

            for ty in range(min_y, max_y + 1):
                for tx in range(min_x, max_x + 1):
                    candidates = possible_blocks_for(tx, ty)
                    if not candidates:
                        raise Contradiction(f"No kxk block possible at top-left {(tx, ty)}")

                    # For each cell in this block, restrict its domain to chars that appear
                    # in that position among candidate blocks.
                    for dy in range(k):
                        for dx in range(k):
                            allowed_chars = {b[dy][dx] for b in candidates}
                            cell = wave[ty + dy][tx + dx]
                            new_cell = cell.intersection(allowed_chars)
                            if new_cell != cell:
                                wave[ty + dy][tx + dx] = new_cell
                                queue.append((tx + dx, ty + dy))

    # Initialize by propagating nothing yet (all domains full)

    # Main collapse loop: pick cell with minimum entropy (>1)
    for _ in range(out_w * out_h):
        best: Optional[Coord] = None
        best_e = float("inf")
        for y in range(out_h):
            for x in range(out_w):
                if collapsed[y][x] is not None:
                    continue
                opts = wave[y][x]
                if len(opts) <= 1:
                    continue
                local_weights = local_char_distribution_for_cell(x, y)
                e = shannon_entropy_labels(set(local_weights.keys()), local_weights) + rng.random() * 1e-6
                if e < best_e:
                    best_e = e
                    best = (x, y)

        if best is None:
            break

        bx, by = best
        opts = list(wave[by][bx])
        if not opts:
            raise Contradiction(f"No options at {(bx, by)}")
        local_weights = local_char_distribution_for_cell(bx, by)
        # ensure we only sample from opts
        sample_items = [ch for ch in opts if ch in local_weights]
        if not sample_items:
            sample_items = opts
            local_weights = {ch: float(global_char_weights.get(ch, 1.0)) for ch in sample_items}
        chosen = weighted_choice_str(rng, sample_items, local_weights)
        wave[by][bx] = {chosen}
        collapsed[by][bx] = chosen
        propagate([(bx, by)])

    # finalize
    out: List[str] = []
    for y in range(out_h):
        row_chars: List[str] = []
        for x in range(out_w):
            if collapsed[y][x] is None:
                opts = wave[y][x]
                if not opts:
                    raise Contradiction(f"No options at {(x, y)}")
                local_weights = local_char_distribution_for_cell(x, y)
                sample_items = [ch for ch in opts if ch in local_weights]
                if not sample_items:
                    sample_items = list(opts)
                    local_weights = {ch: float(global_char_weights.get(ch, 1.0)) for ch in sample_items}
                collapsed[y][x] = weighted_choice_str(rng, sample_items, local_weights)
            row_chars.append(collapsed[y][x] or "-")
        out.append("".join(row_chars))

    # Final validation: every k×k block must be in allowed_blocks
    for y in range(0, out_h - k + 1):
        for x in range(0, out_w - k + 1):
            blk = tuple(out[y + dy][x : x + k] for dy in range(k))
            if blk not in model.allowed_blocks:
                raise Contradiction("Final output violated kxk constraint")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ASCII Mario maps using Wave Function Collapse (WFC).")
    parser.add_argument(
        "--reference",
        action="append",
        required=True,
        help="Reference map path(s). You can pass multiple times to learn from multiple maps.",
    )
    parser.add_argument("--out-dir", default= "scenarios/mario/levels", help="Output directory to save generated maps")
    parser.add_argument("--count", type=int, default=25, help="Number of maps to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--square-size",
        type=int,
        default=2,
        help="Square constraint size k (k×k). Default: 2. No rotations are used.",
    )
    parser.add_argument(
        "--same-size",
        action="store_true",
        help="Generate maps with the same width/height as the first reference map",
    )
    parser.add_argument("--width", type=int, default=60, help="Output width (ignored if --same-size)")
    parser.add_argument("--height", type=int, default=16, help="Output height (ignored if --same-size)")
    parser.add_argument(
        "--fenced",
        action="store_true",
        help="Wrap each output file in a ```ascii fenced code block (optional)",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=100,
        help="Max attempts per map (WFC can contradict; we retry)",
    )

    args = parser.parse_args()

    rng = random.Random(args.seed)

    ref_paths = [Path(p) for p in args.reference]
    ref_grids = [read_ascii_map(p) for p in ref_paths]

    if args.same_size:
        out_h = len(ref_grids[0])
        out_w = len(ref_grids[0][0])
    else:
        out_w = int(args.width)
        out_h = int(args.height)

    k = int(args.square_size)

    model = build_model(ref_grids, k=k)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(1, int(args.count) + 1):
        last: Optional[Exception] = None
        for attempt in range(1, int(args.attempts) + 1):
            try:
                local_rng = random.Random(rng.randint(0, 2**31 - 1))
                lines = wfc_generate(model, out_w=out_w, out_h=out_h, rng=local_rng)
                out_path = out_dir / f"map_{idx:03d}.txt"
                write_ascii_map(out_path, lines, fenced=bool(args.fenced))
                print(f"Wrote {out_path}")
                last = None
                break
            except Exception as e:
                last = e
        if last is not None:
            raise SystemExit(f"Failed to generate map {idx:03d} after {args.attempts} attempts: {last}")


if __name__ == "__main__":
    main()
