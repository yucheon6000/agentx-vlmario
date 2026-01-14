<table width="300%">
<tr>
<td width="200" align="center">
<img src="assets/vlmario-logo.png" alt="VLMario 로고" width="180"/>
</td>
<td width="1300" align="center">

<h1>VLMario 벤치마크</h1>

**마리오 레벨 생성 및 평가를 위한 시각-언어 모델(VLM) 벤치마크**

[개요](#개요) • [아키텍처](#아키텍처) • [설치 방법](#설치-방법) • [빠른 시작](#빠른-시작) • [맵 생성기](#맵-생성기) • [평가](#평가) • [ASCII 참조 가이드](#ascii-참조-가이드)

</td>
</tr>
</table>

---

## 개요

VLMario는 AI 에이전트가 Super Mario Bros. 스타일의 플레이 가능한 레벨을 얼마나 잘 생성하는지 평가하기 위한 오픈 벤치마크 프레임워크입니다. 이 벤치마크는 시각-언어 모델(VLM)을 활용하여 게임플레이 시뮬레이션 비디오를 바탕으로 생성된 레벨을 정밀하게 평가합니다.

### 주요 기능

- **자동 평가 파이프라인**: 맵 생성 → 게임플레이 시뮬레이션 → VLM을 이용한 자동 평가
- **다차원 채점**: 8가지 평가 기준 (구성항목, 개연성, 완결성, 심미성, 독창성, 공정성, 재미, 난이도)
- **Top-K 집계**: 25개의 맵을 평가하고 상위 5개의 점수를 사용하여 최종 점수 산출
- **확장 가능한 구조**: 사용자 정의 맵 생성기를 쉽게 통합 가능
- **A2A 프로토콜 지원**: 에이전트 간 통신 표준 규격 준수

## 아키텍처

<p align="center">
  <img src="assets/structure.png" alt="VLMario 아키텍처" width="700"/>
</p>

VLMario 벤치마크는 두 개의 주요 컴포넌트로 구성됩니다:

1. **맵 디자이너 (Purple Agent)**: ASCII 기반의 마리오 레벨을 생성합니다.
2. **맵 평가자 (Green Agent)**: 다음과 같은 과정을 통해 평가를 조율합니다:
   - 디자이너에게 맵 생성을 요청
   - `PlayAstar.jar`를 사용하여 A* 시뮬레이션 실행
   - 게임플레이 비디오 녹화
   - Gemini VLM을 사용하여 비디오 평가
   - 여러 맵에 걸친 점수 집계

## 설치 방법

### 사전 요구 사항

- **Docker** (필수)
- **Google API Key** (Gemini 모델 사용을 위함)
- **인터넷 연결** (이미지 빌드 및 API 통신용)

### 1단계: 저장소 클론

```bash
git clone <repository-url>
cd agentx-vlmario
```

### 2단계: 이미지 빌드

Dockerfile을 사용하여 벤치마크 실행 환경을 구축합니다. 자바, ffmpeg, 필요한 파이썬 패키지들이 자동으로 포함됩니다.

```bash
docker build -t vlmario .
```

### 3단계: 환경 변수 설정

```bash
cp sample.env .env
```

`.env` 파일을 열고 Google API 키를 입력합니다:

```env
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=your_google_api_key_here
```

### 4단계: 실행 준비 완료
준비가 끝났습니다. 이제 아래의 [빠른 시작](#빠른-시작) 섹션을 따라 실행하세요.

## 빠른 시작 (권장)

생성된 맵, 비디오, 평가 결과를 내 로컬 PC에서 즉시 확인하려면 **볼륨 마운트(Volume Mount)** 옵션을 사용하세요. 이 방식은 이미지를 다시 빌드하지 않고도 코드 변경 사항을 실시간으로 반영할 수 있게 해줍니다:

```bash
# Windows PowerShell 기준
docker run -it --env-file .env -v ${PWD}:/app -v /app/.venv vlmario

# macOS/Linux 기준
docker run -it --env-file .env -v $(pwd):/app -v /app/.venv vlmario
```

실행 시 다음 과정이 내부적으로 수행됩니다:
1. 맵 평가자(Green Agent) 서버 시작
2. 맵 디자이너(Purple Agent) 서버 시작
3. 평가자가 디자이너에게 25개의 맵을 요청
4. 각 맵은 게임플레이 시뮬레이션을 통해 평가
5. 모든 결과(맵, 비디오, JSON)를 `outputs/` 폴더에 저장
6. 상위 5개 맵을 기준으로 최종 점수 리포트

### 옵션 2: 미리 생성된 맵 파일 사용

`scenarios/mario/levels/` 디렉토리에 미리 생성된 맵 파일을 넣고 평가를 진행할 수 있습니다:

1. 맵 파일(`.txt` 형식)을 생성합니다.
2. 생성한 파일을 `scenarios/mario/levels/`에 넣습니다.
3. 벤치마크를 실행합니다.

### 옵션 3: 커스텀 맵 생성기 사용

1. 본인만의 생성기를 구현합니다 ([맵 생성기](#맵-생성기) 섹션 참조).
2. `run_generator.sh`가 본인의 생성기를 실행하도록 설정합니다.
3. `scenarios/mario/scenario.toml` 파일에서 `pre_cmd` 주석을 해제합니다:

```toml
[green_agent]
endpoint = "http://127.0.0.1:9100"
pre_cmd = "bash scenarios/mario/run_generator.sh"
cmd = "python scenarios/mario/mario_map_evaluator.py --host 127.0.0.1 --port 9100"
```

### 추가 옵션

- **평가 중 로그 표시**:
  명령어 끝에 `--show-logs`를 추가하세요:
  ```bash
  docker run -it --env-file .env -v ${PWD}:/app -v /app/.venv vlmario --show-logs
  ```

- **에이전트 서버만 시작 (디버깅용)**:
  ```bash
  docker run -it --env-file .env -v ${PWD}:/app -v /app/.venv vlmario --serve-only
  ```

## 맵 생성기

VLMario는 커스텀 맵 생성기를 지원합니다. 생성기는 ASCII 맵 파일을 `.txt` 형식으로 `scenarios/mario/levels/` 디렉토리에 출력해야 합니다.

### 생성기 요구 사항

1. **출력 형식**: 일반 텍스트(Plain text) ASCII 맵
2. **출력 위치**: `scenarios/mario/levels/` 디렉토리
3. **파일명**: 자유로운 `.txt` 파일명 (예: `map_001.txt`, `level_1.txt`)
4. **맵 형식**: 아래의 [ASCII 참조 가이드](#ascii-참조-가이드)를 따를 것

### 예시 생성기

두 가지 참조 구현을 제공합니다:

#### 1. LLM 기반 생성기 (`generate_llm.py`)

프롬프팅을 통해 거대 언어 모델이 맵을 생성하도록 합니다:

```bash
uv run python scenarios/mario/generate_llm.py --model gemini-2.0-flash --count 25
```

옵션:
- `--model`: 사용할 LLM 모델명 (기본값: gemini-2.0-flash)
- `--count`: 생성할 맵의 개수 (기본값: 25)
- `--seed`: 재현성을 위한 랜덤 시드

#### 2. WFC(Wave Function Collapse) 생성기 (`generate_wfc.py`)

패턴 학습 기반의 절차적 생성을 사용합니다:

```bash
uv run python scenarios/mario/generate_wfc.py \
    --reference scenarios/mario/levels/text_level_0.txt \
    --out-dir scenarios/mario/levels \
    --count 25
```

옵션:
- `--reference`: 패턴 학습을 위한 참조 맵 파일
- `--out-dir`: 생성된 맵이 저장될 디렉토리
- `--count`: 생성할 맵의 개수
- `--same-size`: 참조 맵과 동일한 크기로 생성
- `--seed`: 재현성을 위한 랜덤 시드

### 커스텀 생성기 제작하기

커스텀 생성기를 만들려면 다음 절차를 따르세요:

1. `scenarios/mario/prompts/map_ascii_guide.md`의 ASCII 가이드를 숙지합니다.
2. 다음 요소를 포함하는 맵을 생성합니다:
   - `M`: 마리오 시작 위치
   - `F`: 종료 지점(깃발) 위치
   - 일정한 행 너비 (빈 공간은 `-`로 채움)
3. 출력 파일을 `scenarios/mario/levels/`에 저장합니다.
4. `run_generator.sh`를 본인의 생성기 명령어로 업데이트합니다.

예시 `run_generator.sh`:

```bash
#!/bin/bash
python your_generator.py --output-dir scenarios/mario/levels --count 25
```

## 평가

### 채점 시스템

맵은 8가지 기준에 따라 1-20점 사이의 점수로 평가되며, 각 기준은 7점 리커트 척도(Likert scale)로 측정됩니다:

| 평가 기준 | 설명 |
|-----------|-------------|
| **구성항목 (Composition)** | 마리오의 필수 요소(시작, 종료, 발판, 적)가 존재하는지 여부 |
| **개연성 (Probability)** | 배치가 오리지널 마리오의 논리적 제약 사항을 따르는지 여부 |
| **완결성 (Completeness)** | 구성 요소들이 플레이어의 전략적 의사결정에 영향을 주는지 여부 |
| **심미성 (Aesthetics)** | 시각적 균형과 전반적인 미적 완성도 |
| **독창성 (Originality)** | 고유하거나 흔치 않은 구조적 아이디어의 존재 여부 |
| **공정성 (Fairness)** | 불공평하거나 갑작스러운, 예측 불가능한 위험 요소의 배제 여부 |
| **재미 (Fun)** | 레벨이 즐겁게 플레이될 수 있을 것으로 보이는지 여부 |
| **난이도 (Difficulty)** | 전반적으로 인지되는 난이도 수준 |

### 집계 방법

- **총 평가 맵 수**: 25개
- **최종 점수에 반영될 상위 K개**: 5개 (설정 가능)
- **최종 점수**: 상위 5개 맵의 평균 점수

이 방식은 일관성을 보상함과 동시에 실험적인 시도를 허용합니다.

### 실패 및 페널티 규정

정상적인 평가가 불가능한 경우, 에이전트의 성능 성취도를 정확히 반영하기 위해 해당 회차는 **최저점(1점)**으로 처리됩니다:

- **통신 실패 및 타임아웃**: 에이전트가 응답하지 않거나 요청이 중단된 경우.
- **맵 추출 실패**: 응답에서 ASCII 맵 형식을 찾아낼 수 없는 경우.
- **시뮬레이션 실패**: 생성된 맵이 플레이 불가능하여 게임플레이 영상이 생성되지 않는 경우.
- **평가 에러**: LLM 서비스 장애 등으로 평가 결과를 얻지 못한 경우.

*위 경우 모두 총점 1점 및 모든 세부 항목 점수(composition 등)가 1점으로 기록되며, 결과 JSON에 원인이 명시됩니다.*

### 설정

`scenarios/mario/scenario.toml`에서 평가 파라미터를 수정할 수 있습니다:

```toml
[config]
num_maps = 25                    # 총 평가할 맵 수
# top_k = 5                      # 최종 점수 산출을 위한 상위 맵 수
jar_output_dir = "./"            # 게임플레이 비디오 저장 디렉토리
jar_output_name_template = "{role}_gameplay_{ts}_{map_idx}.mp4"
```

### 결과물

평가가 완료되면 모든 결과 파일이 `outputs/` 디렉토리에 저장됩니다. 동일한 세션에서 생성된 모든 파일은 동일한 타임스탬프를 공유하여 추적이 용이합니다:

- **ASCII 맵**: `YYYYMMDD_HHMMSS_{idx}_map.txt`
- **게임플레이 비디오**: `YYYYMMDD_HHMMSS_{idx}_video.mp4`
- **개별 결과 (JSON)**: `YYYYMMDD_HHMMSS_{idx}_result.json`
- **최정 요약 결과**: `YYYYMMDD_HHMMSS_total_result.json`

## ASCII 참조 가이드

### 맵 형식 요구 사항

- **규격**: 가로 70-90자 권장
- **행 일관성**: 모든 행은 동일한 길이여야 함 (빈 공간은 `-`로 채움)
- **필수 요소**: `M` (시작) 및 `F` (종료)
- **플레이 가능성**: 60초 이내에 클리어 가능해야 함

### ASCII 타일 가이드

#### 레벨 경계
| 문자 | 설명 |
|-----------|-------------|
| `M` | 마리오 시작 위치 |
| `F` | 마리오 종료 지점 (깃발) |
| `-` 또는 공백 | 빈 공간 (공기) |

#### 지형 및 블록
| 문자 | 설명 |
|-----------|-------------|
| `X` | 지면 (단단한 바닥) |
| `#` | 피라미드 블록 (계단/장식용) |
| `S` | 일반 벽돌 (파괴 가능) |
| `C` | 코인 벽돌 |
| `L` | 1-Up 벽돌 |
| `U` | 버섯 벽돌 |
| `D` | 사용된 블록 (이미 친 블록) |
| `%` | 발판 (아래에서 통과 가능) |
| `|` | 발판 배경 |

#### 물음표 블록
| 문자 | 설명 |
|-----------|-------------|
| `?` 또는 `@` | 버섯 물음표 블록 |
| `Q` or `!` | 코인 물음표 블록 |
| `1` | 투명 1-Up 블록 |
| `2` | 투명 코인 블록 |

#### 아이템
| 문자 | 설명 |
|-----------|-------------|
| `o` | 수집 가능한 코인 |

#### 파이프(토관)
| 문자 | 설명 |
|-----------|-------------|
| `t` | 빈 파이프 |
| `T` | 꽃 파이프 (뻐끔꽃 포함) |
| `<` `>` | 파이프 상단 (좌/우) |
| `[` `]` | 파이프 몸통 (좌/우) |

#### 킬러 대포
| 문자 | 설명 |
|-----------|-------------|
| `*` | 킬러 대포 |
| `B` | 킬러 대포 머리 |
| `b` | 킬러 대포 몸통 |

#### 적 캐릭터
| 문자 | 설명 |
|-----------|-------------|
| `E` 또는 `g` | 굼바 |
| `G` | 날개 달린 굼바 |
| `k` | 초록 엉금엉금 |
| `K` | 날개 달린 초록 엉금엉금 |
| `r` | 빨간 엉금엉금 |
| `R` | 날개 달린 빨간 엉금엉금 |
| `y` | 가시돌이 |
| `Y` | 날개 달린 가시돌이 |

### 맵 예시

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

전체 ASCII 가이드는 `scenarios/mario/prompts/map_ascii_guide.md`에서 확인하세요.

## 프로젝트 구조

```
agentx-vlmario/
├── scenarios/
│   └── mario/
│       ├── scenario.toml           # 벤치마크 설정 파일
│       ├── mario_map_evaluator.py  # Green 에이전트 (평가자)
│       ├── mario_map_designer.py   # Purple 에이전트 (LLM 디자이너)
│       ├── generate_llm.py         # LLM 기반 맵 생성 스크립트
│       ├── generate_wfc.py         # WFC 기반 맵 생성 스크립트
│       ├── run_generator.sh        # 커스텀 생성기 실행 스크립트
│       ├── PlayAstar.jar           # 게임플레이 시뮬레이터
│       ├── levels/                 # 맵 파일 디렉토리
│       │   ├── test_level_1.txt
│       │   ├── test_level_2.txt
│       │   └── ...
│       ├── prompts/                # 프롬프트 템플릿
│       │   ├── system_prompt.md
│       │   ├── map_ascii_guide.md
│       │   ├── map_request.md
│       │   └── initial_criterion_prompt.md
│       └── img/                    # 시뮬레이션용 게임 에셋
├── src/
│   └── agentbeats/                 # 코어 프레임워크
│       ├── run_scenario.py         # 시나리오 실행기
│       ├── green_executor.py       # Green 에이전트 실행기
│       ├── models.py               # 데이터 모델
│       └── ...
├── assets/
│   ├── vlmario-logo.png            # 벤치마크 로고
│   └── structure.png               # 아키텍처 다이어그램
├── pyproject.toml                  # 프로젝트 의존성 설정
├── sample.env                      # 환경 설정 템플릿
└── README.md                       # 영문 리드미 파일
```

## Docker 지원

Docker를 사용하여 빌드 및 실행하는 방법입니다:

```bash
# 이미지 빌드
docker build -t vlmario .

# 컨테이너 실행
docker run -it --env-file .env vlmario
```

## 문제 해결 (Troubleshooting)

### 주요 문제 상황

1. **Java를 찾을 수 없음**
   ```
   Error: PlayAstar.jar execution failed
   ```
   해결책: Java Runtime Environment (JRE)를 설치하세요.

2. **API Key 오류**
   ```
   Error: Google API authentication failed
   ```
   해결책: `.env` 파일의 `GOOGLE_API_KEY`를 다시 확인하세요.

3. **맵이 생성되지 않음**
   ```
   Warning: Failed to extract ASCII map from response
   ```
   해결책: 생성기의 출력 형식이 ASCII 명세와 일치하는지 확인하세요.

4. **비디오 생성 실패**
   ```
   Error: No video generated (simulation likely failed)
   ```
   해결책: 맵이 유효한지 확인하고 `PlayAstar.jar`에 실행 권한이 있는지 확인하세요.

### 디버그 모드

컨테이너 실행 시 상세 로그를 확인하려면 명령어 끝에 `--show-logs`를 추가하세요:

```bash
docker run -it --env-file .env -v ${PWD}:/app -v /app/.venv vlmario --show-logs
```

## 기여하기 (Contributing)

여러분의 기여를 환영합니다! 다음 항목에 대한 기여 방법을 확인해 보세요:
- 새로운 맵 생성기 추가
- 평가 기준 개선
- 시뮬레이션 파이프라인 고도화

## 라이선스

이 프로젝트는 오픈 소스입니다. 자세한 내용은 LICENSE 파일을 참조하세요.

## 감사 인사 (Acknowledgments)

- [AgentBeats](https://github.com/rdi-foundation/agentbeats-tutorial) 프레임워크를 기반으로 구축되었습니다.
- 에이전트 간 통신을 위해 [A2A 프로토콜](https://a2a-protocol.org/)을 사용합니다.
- 시각-언어 평가를 위해 Google Gemini를 활용합니다.
