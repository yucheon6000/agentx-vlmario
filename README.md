<table width="300%">
<tr>
<td width="200" align="center">
<img src="assets/vlmario-logo.png" alt="VLMario Logo" width="180"/>
</td>
<td width="1300" align="center">

<h1>VLMario Benchmark</h1>

**A Vision-Language Model Benchmark for Mario Level Generation and Evaluation**

[Overview](#overview) • [Architecture](#architecture) • [Installation](#installation) • [Quick Start](#quick-start) • [Map Generators](#map-generators) • [Evaluation](#evaluation) • [ASCII Reference](#ascii-reference)

</td>
</tr>
</table>

---

## Overview

VLMario is an open benchmark framework for evaluating AI agents' ability to generate playable Super Mario Bros.-style levels. The benchmark leverages Vision-Language Models (VLMs) to assess generated levels based on gameplay simulation videos.

### Key Features

- **Automated Evaluation Pipeline**: Generate maps → Simulate gameplay → Evaluate with VLM
- **Multi-dimensional Scoring**: 8 evaluation criteria (composition, probability, completeness, aesthetics, originality, fairness, fun, difficulty)
- **Top-K Aggregation**: Evaluates 25 maps and uses top 5 for final scoring
- **Extensible Architecture**: Easy to integrate custom map generators
- **A2A Protocol Support**: Compatible with agent-to-agent communication standards

## Architecture

<p align="center">
  <img src="assets/structure.png" alt="VLMario Architecture" width="700"/>
</p>

The VLMario benchmark consists of two main components:

1. **Map Designer (Purple Agent)**: Generates ASCII-based Mario levels
2. **Map Evaluator (Green Agent)**: Orchestrates evaluation by:
   - Requesting maps from the designer
   - Running A* simulation using `PlayAstar.jar`
   - Recording gameplay videos
   - Evaluating videos using Gemini VLM
   - Aggregating scores across multiple maps

## Installation

### Prerequisites

- **Docker** (Required)
- **Google API Key** (for Gemini models)
- **Internet Connection** (for building image and API communication)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd agentx-vlmario
```

### Step 2: Build the Image

Build the benchmark execution environment using Docker. It automatically includes Java, ffmpeg, and necessary Python packages.

```bash
docker build -t vlmario .
```

### Step 3: Configure Environment Variables

```bash
cp sample.env .env
```

Edit the `.env` file and add your Google API key:

```env
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=your_google_api_key_here
```

### Step 4: Ready to Run
You're all set. Follow the [Quick Start](#quick-start) section below.

### Quick Start (Recommended)

To see the generated maps, videos, and evaluation results on your local machine, use the **Volume Mount** option. This also allows you to see code changes in real-time without rebuilding the image:

```bash
# For Windows PowerShell
docker run -it --env-file .env -v ${PWD}:/app -v /app/.venv vlmario

# For macOS/Linux
docker run -it --env-file .env -v $(pwd):/app -v /app/.venv vlmario
```

This will automatically:
1. Start the Map Evaluator (Green Agent)
2. Start the Map Designer (Purple Agent)
3. Request 25 maps from the designer
4. Evaluate each map using gameplay simulation
5. Save all results (maps, videos, JSON) to the `outputs/` folder
6. Report final scores based on top 5 maps

### Option 2: Use Pre-generated Maps

You can also place pre-generated map files in the `scenarios/mario/levels/` directory and run the evaluation:

1. Generate or create map files (`.txt` format)
2. Place them in `scenarios/mario/levels/`
3. Run the benchmark

### Option 3: Use Custom Map Generator

1. Implement your generator (see [Map Generators](#map-generators))
2. Configure `run_generator.sh` to run your generator
3. Uncomment the `pre_cmd` in `scenarios/mario/scenario.toml`:

```toml
[green_agent]
endpoint = "http://127.0.0.1:9100"
pre_cmd = "bash scenarios/mario/run_generator.sh"
cmd = "python scenarios/mario/mario_map_evaluator.py --host 127.0.0.1 --port 9100"
```

### Additional Options

- **Show logs during evaluation**:
  Add `--show-logs` at the end of the docker command:
  ```bash
  docker run -it --env-file .env -v ${PWD}:/app -v /app/.venv vlmario --show-logs
  ```

- **Start agents only (for debugging)**:
  ```bash
  docker run -it --env-file .env -v ${PWD}:/app -v /app/.venv vlmario --serve-only
  ```

## Map Generators

VLMario supports custom map generators. Your generator must output ASCII map files in `.txt` format to the `scenarios/mario/levels/` directory.

### Generator Requirements

1. **Output Format**: Plain text ASCII maps
2. **Output Location**: `scenarios/mario/levels/` directory
3. **File Naming**: Any `.txt` filename (e.g., `map_001.txt`, `level_1.txt`)
4. **Map Format**: See [ASCII Reference](#ascii-reference) below

### Example Generators

We provide two reference implementations:

#### 1. LLM-based Generator (`generate_llm.py`)

Uses large language models to generate maps through prompting:

```bash
uv run python scenarios/mario/generate_llm.py --model gemini-2.0-flash --count 25
```

Options:
- `--model`: LLM model name (default: gemini-2.0-flash)
- `--count`: Number of maps to generate (default: 25)
- `--seed`: Random seed for reproducibility

#### 2. Wave Function Collapse Generator (`generate_wfc.py`)

Uses procedural generation based on pattern learning:

```bash
uv run python scenarios/mario/generate_wfc.py \
    --reference scenarios/mario/levels/text_level_0.txt \
    --out-dir scenarios/mario/levels \
    --count 25
```

Options:
- `--reference`: Reference map(s) for pattern learning
- `--out-dir`: Output directory for generated maps
- `--count`: Number of maps to generate
- `--same-size`: Generate maps with same dimensions as reference
- `--seed`: Random seed for reproducibility

### Creating Your Own Generator

To create a custom generator:

1. Read the ASCII guide in `scenarios/mario/prompts/map_ascii_guide.md`
2. Generate maps that include:
   - `M`: Mario start position
   - `F`: Exit flag position
   - Consistent row widths (pad with `-`)
3. Save output files to `scenarios/mario/levels/`
4. Update `run_generator.sh` with your generator command

Example `run_generator.sh`:

```bash
#!/bin/bash
python your_generator.py --output-dir scenarios/mario/levels --count 25
```

## Evaluation

### Scoring System

Maps are evaluated on a scale of 1-20 based on 8 criteria, each scored on a 7-point Likert scale:

| Criterion | Description |
|-----------|-------------|
| **Composition** | Whether all essential SMB components exist (start, goal, platforms, enemies) |
| **Probability** | Whether placements follow logical constraints of original SMB |
| **Completeness** | Whether components influence strategic decision-making |
| **Aesthetics** | Visual balance and overall aesthetic appeal |
| **Originality** | Presence of unique or uncommon structural ideas |
| **Fairness** | Avoidance of unfair, sudden, or unpredictable hazards |
| **Fun** | Whether the level appears enjoyable to play |
| **Difficulty** | Overall perceived difficulty |

### Aggregation Method

- **Total Maps Evaluated**: 25
- **Top-K for Final Score**: 5 (configurable)
- **Final Score**: Average of top 5 highest-scoring maps

This approach rewards consistency while allowing for experimental variations.

### Failure Handling & Penalties

If a normal evaluation cannot be performed, the attempt is recorded with a **minimum score (1 point)** to accurately reflect the agent's reliability:

- **Communication Failure & Timeout**: When the agent does not respond or the request is interrupted.
- **Extraction Failure**: When an ASCII map cannot be found/parsed from the agent's response.
- **Simulation Failure**: When the generated map is unplayable and no gameplay video is created.
- **Evaluation Error**: When results cannot be obtained due to LLM service issues.

*In these cases, the total score and all sub-category scores (composition, etc.) are set to 1, and the reason is documented in the result JSON.*

### Configuration

Modify evaluation parameters in `scenarios/mario/scenario.toml`:

```toml
[config]
num_maps = 25                    # Total maps to evaluate
# top_k = 5                      # Number of top maps for final score
jar_output_dir = "./"            # Directory for gameplay videos
jar_output_name_template = "{role}_gameplay_{ts}_{map_idx}.mp4"
```

### Output

After evaluation, all artifacts are saved to the `outputs/` directory. All files within a single session share the same timestamp for easy tracking:

- **ASCII Maps**: `YYYYMMDD_HHMMSS_{idx}_map.txt`
- **Gameplay Videos**: `YYYYMMDD_HHMMSS_{idx}_video.mp4`
- **Individual Results**: `YYYYMMDD_HHMMSS_{idx}_result.json`
- **Aggregated Summary**: `YYYYMMDD_HHMMSS_total_result.json`

## ASCII Reference

### Map Format Requirements

- **Dimensions**: 70-90 characters wide recommended
- **Row Consistency**: All rows must be the same length (pad with `-`)
- **Required Elements**: `M` (start) and `F` (exit)
- **Playability**: Level should be completable within 60 seconds

### ASCII Tile Guide

#### Level Boundaries
| Character | Description |
|-----------|-------------|
| `M` | Mario Start Position |
| `F` | Mario Exit (Flag) |
| `-` or space | Empty space (air) |

#### Terrain & Blocks
| Character | Description |
|-----------|-------------|
| `X` | Ground (solid floor) |
| `#` | Pyramid Block (stairs/decorative) |
| `S` | Normal Brick (breakable) |
| `C` | Coin Brick |
| `L` | 1-Up Brick |
| `U` | Mushroom Brick |
| `D` | Used Block (already hit) |
| `%` | Platform (jump-through) |
| `\|` | Platform Background |

#### Question Blocks
| Character | Description |
|-----------|-------------|
| `?` or `@` | Mushroom Question Block |
| `Q` or `!` | Coin Question Block |
| `1` | Invisible 1-Up Block |
| `2` | Invisible Coin Block |

#### Items
| Character | Description |
|-----------|-------------|
| `o` | Collectible Coin |

#### Pipes
| Character | Description |
|-----------|-------------|
| `t` | Empty Pipe |
| `T` | Flower Pipe (with Piranha Plant) |
| `<` `>` | Pipe Top (left/right) |
| `[` `]` | Pipe Body (left/right) |

#### Bullet Bill Cannons
| Character | Description |
|-----------|-------------|
| `*` | Bullet Bill Cannon |
| `B` | Bullet Bill Head |
| `b` | Bullet Bill Body |

#### Enemies
| Character | Description |
|-----------|-------------|
| `E` or `g` | Goomba |
| `G` | Winged Goomba |
| `k` | Green Koopa |
| `K` | Winged Green Koopa |
| `r` | Red Koopa |
| `R` | Winged Red Koopa |
| `y` | Spiny |
| `Y` | Winged Spiny |

### Example Map

```ascii
----------------------------------------------------------------------------------------------------
----------------------------------------------Q---Q---Q---------------------------------------------
----------------------------------------------------------------------------------------------------
--------------------------E--------------------------------------------E----------------------------
XXXXXXXXXXXX----------XXXXXXXX--------<>--------XXXXXXXXX--------<>-----------XXXXXXXXX----F--------
XXXXXXXXXXXX----------XXXXXXXX--------[]--------XXXXXXXXX--------[]-----------XXXXXXXXX---XXX-------
XXXXXXXXXXXX----------XXXXXXXX--------[]--------XXXXXXXXX--------[]-----------XXXXXXXXX--XXXXX------
M-XXXXXXXXXX----------XXXXXXXX-------XXXX-------XXXXXXXXX-------XXXX----------XXXXXXXXX-XXXXXXX-----
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

For the complete ASCII guide, see `scenarios/mario/prompts/map_ascii_guide.md`.

## Project Structure

```
agentx-vlmario/
├── scenarios/
│   └── mario/
│       ├── scenario.toml           # Benchmark configuration
│       ├── mario_map_evaluator.py  # Green agent (evaluator)
│       ├── mario_map_designer.py   # Purple agent (LLM designer)
│       ├── generate_llm.py         # LLM-based map generator
│       ├── generate_wfc.py         # WFC-based map generator
│       ├── run_generator.sh        # Custom generator script
│       ├── PlayAstar.jar           # Gameplay simulator
│       ├── levels/                 # Map files directory
│       │   ├── test_level_1.txt
│       │   ├── test_level_2.txt
│       │   └── ...
│       ├── prompts/                # Prompt templates
│       │   ├── system_prompt.md
│       │   ├── map_ascii_guide.md
│       │   ├── map_request.md
│       │   └── initial_criterion_prompt.md
│       └── img/                    # Game assets for simulation
├── src/
│   └── agentbeats/                 # Core framework
│       ├── run_scenario.py         # Scenario runner
│       ├── green_executor.py       # Green agent executor
│       ├── models.py               # Data models
│       └── ...
├── assets/
│   ├── vlmario-logo.png            # Benchmark logo
│   └── structure.png               # Architecture diagram
├── pyproject.toml                  # Project dependencies
├── sample.env                      # Environment template
└── README.md                       # This file
```

## Docker Support

Build and run using Docker:

```bash
# Build the image
docker build -t vlmario .

# Run the container
docker run -it --env-file .env vlmario
```

## Troubleshooting

### Common Issues

1. **Java not found**
   ```
   Error: PlayAstar.jar execution failed
   ```
   Solution: Install Java Runtime Environment (JRE)

2. **API Key error**
   ```
   Error: Google API authentication failed
   ```
   Solution: Verify `GOOGLE_API_KEY` in `.env` file

3. **No maps generated**
   ```
   Warning: Failed to extract ASCII map from response
   ```
   Solution: Check generator output format matches ASCII specification

4. **Video generation failed**
   ```
   Error: No video generated (simulation likely failed)
   ```
   Solution: Ensure map is valid and `PlayAstar.jar` has execute permissions

### Debug Mode

Run with detailed logs in the container:

```bash
docker run -it --env-file .env vlmario --show-logs
```

## Contributing

We welcome contributions! Please see our contributing guidelines for:
- Adding new map generators
- Improving evaluation criteria
- Enhancing the simulation pipeline

## License

This project is open source. See LICENSE file for details.

## Acknowledgments

- Built on the [AgentBeats](https://github.com/rdi-foundation/agentbeats-tutorial) framework
- Uses the [A2A Protocol](https://a2a-protocol.org/) for agent communication
- Powered by Google Gemini for vision-language evaluation

