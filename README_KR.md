<table width="300%">
<tr>
<td width="200" align="center">
<img src="assets/vlmario-logo.png" alt="VLMario 로고" width="180"/>
</td>
<td width="1300" align="center">

<h1>VLMario 벤치마크</h1>

**마리오 레벨 생성 및 평가를 위한 비전-언어 모델(Vision-Language Model) 벤치마크**

[개요](#개요) • [아키텍처](#아키텍처) • [설치](#설치) • [빠른 시작](#빠른-시작) • [맵 생성기](#맵-생성기) • [평가](#평가) • [ASCII 참조](#ascii-참조)

</td>
</tr>
</table>

---

## 개요

VLMario는 AI 에이전트가 플레이 가능한 슈퍼 마리오 브라더스 스타일의 레벨을 생성하는 능력을 평가하기 위한 오픈 벤치마크 프레임워크입니다. 이 벤치마크는 비전-언어 모델(VLM)을 활용하여 게임 플레이 시뮬레이션 영상을 기반으로 생성된 레벨을 평가합니다.

### 주요 기능

- **자동화된 평가 파이프라인**: 맵 생성 → 게임 플레이 시뮬레이션 → VLM으로 평가
- **다차원 채점**: 8가지 평가 기준 (구성 요소, 개연성, 완결성, 심미성, 독창성, 공정성, 재미, 난이도)
- **Top-K 집계**: 25개의 맵을 평가하고 상위 5개를 사용하여 최종 점수 산출

## 아키텍처

<p align="center">
  <img src="assets/structure.png" alt="VLMario 아키텍처" width="700"/>
</p>

VLMario 벤치마크는 두 가지 주요 구성 요소로 이루어져 있습니다:

1. **맵 디자이너 (Purple Agent)**: ASCII 기반의 마리오 레벨 생성
2. **맵 평가자 (Green Agent)**: 평가를 조율하는 역할을 하며 다음과 같은 작업을 수행합니다:
   - 디자이너에게 맵 요청
   - [Mario-AI-Framework](https://github.com/shyamsn97/Mario-AI-Framework)를 사용하여 A* 시뮬레이션 실행
   - 게임 플레이 영상 녹화
   - Gemini VLM을 사용하여 영상 기반의 맵 평가
   - 여러 맵의 점수 집계

## 설치

### 필수 조건

- **Docker** (필수)
- **Google API Key** (Gemini 모델용)

### 1단계: 저장소 복제

```bash
git clone https://github.com/GIST-CILab/agentx-vlmario.git
cd agentx-vlmario
```

### 2단계: 이미지 빌드

Docker를 사용하여 벤치마크 실행 환경을 빌드합니다. Java, ffmpeg 및 필요한 Python 패키지가 자동으로 포함됩니다.

```bash
docker build -t vlmario .
```

### 3단계: 환경 변수 설정

루트 디렉터리에 `.env` 파일을 생성하고 다음 내용을 추가합니다:

```env
# 필수 설정
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_API_KEY=your_google_api_key_here

# 선택 설정
# OpenRouter 구성
USE_OPEN_ROUTER=FALSE   # 맵 평가자의 모델을 OpenRouter로 변경
OPEN_ROUTER_API_KEY=your_openrouter_api_key_here

# 평가자 (Green Agent) 구성
# 평가 모델의 창의성 제어 (기본값: 0.0)
TEMPERATURE=0.0
```

### 4단계: 실행

로컬 머신에서 생성된 맵, 비디오 및 평가 결과를 확인하려면 **볼륨 마운트** 옵션을 사용하세요. 이를 통해 이미지를 다시 빌드하지 않고도 코드 변경 사항을 실시간으로 확인할 수 있습니다:

```bash
# Windows PowerShell의 경우
docker run -it --env-file .env -v ${PWD}:/app -v /app/.venv vlmario

# macOS/Linux의 경우
docker run -it --env-file .env -v $(pwd):/app -v /app/.venv vlmario

# 추가 옵션
--show-logs: 평가 중 로그 표시
```

이 명령어를 실행하면 자동으로 다음이 수행됩니다:
1. 맵 평가자 (Green Agent) 시작
2. 맵 디자이너 (Purple Agent) 시작
3. 디자이너에게 25개의 맵 요청
4. 게임 플레이 시뮬레이션을 사용하여 각 맵 평가
5. 모든 결과(맵, 비디오, JSON)를 `outputs/` 폴더에 저장
6. 상위 5개 맵을 기반으로 최종 점수 보고

## 평가

### 채점 시스템

맵은 8가지 기준에 따라 1-20점 척도로 평가되며, 각 항목은 7점 리커트 척도로 채점됩니다:

| 기준 | 설명 |
|-----------|-------------|
| **구성 요소 (Composition)** | 필수 Super Mario Bros 구성 요소(시작, 목표, 플랫폼, 적)가 존재하는지 여부 |
| **개연성 (Probability)** | 배치가 원작 Super Mario Bros의 논리적 제약을 따르는지 여부 |
| **완결성 (Completeness)** | 구성 요소가 전략적 의사 결정에 영향을 미치는지 여부 |
| **심미성 (Aesthetics)** | 시각적 균형과 전반적인 미적 매력 |
| **독창성 (Originality)** | 독특하거나 흔치 않은 구조적 아이디어의 존재 여부 |
| **공정성 (Fairness)** | 불공정하거나 갑작스럽거나 예측할 수 없는 위험 요소 회피 여부 |
| **재미 (Fun)** | 레벨이 플레이하기 즐거워 보이는지 여부 |
| **난이도 (Difficulty)** | 전반적으로 인지되는 난이도 |
|||
| **총점 (Total Score)** | Gemini가 모든 항목을 종합적으로 고려하여 판단한 점수 |

### 집계 방법

- **평가된 총 맵 수**: 25개
- **최종 점수용 Top-K**: 5개
- **최종 점수**: 최고 점수를 받은 상위 5개 맵의 평균

이 방식은 일관성을 보상하면서도 실험적인 변형을 허용합니다.

### 실패 처리 및 페널티

정상적인 평가를 수행할 수 없는 경우, 에이전트의 신뢰성을 정확하게 반영하기 위해 **최소 점수(1점)**로 기록됩니다:

- **통신 실패 및 시간 초과**: 에이전트가 응답하지 않거나 요청이 중단된 경우.
- **추출 실패**: 에이전트의 응답에서 ASCII 맵을 찾거나 파싱 할 수 없는 경우.
- **시뮬레이션 실패**: 생성된 맵을 플레이할 수 없어 게임 플레이 영상이 생성되지 않는 경우.
- **평가 오류**: LLM 서비스 문제로 인해 결과를 얻을 수 없는 경우. (5번 연속 실패 시)

*이러한 경우 총점 및 모든 하위 범주 점수(구성 요소 등)는 1로 설정되며, 원인은 결과 JSON에 기록됩니다.*

### 구성

`scenarios/mario/scenario.toml`에서 평가 매개변수를 수정합니다:

```toml
[config]
num_maps = 25                    # 평가할 총 맵 수
top_k = 5                        # 최종 점수용 상위 맵 수
jar_output_dir = "./"            # 게임 플레이 영상 디렉터리
```

### 출력

평가 후 모든 결과물은 `outputs/` 디렉터리에 저장됩니다. 단일 세션 내의 모든 파일은 쉬운 추적을 위해 동일한 타임스탬프를 공유합니다:

- **ASCII 맵**: `YYYYMMDD_HHMMSS_{idx}_map.txt`
- **게임 플레이 영상**: `YYYYMMDD_HHMMSS_{idx}_video.mp4`
- **개별 결과**: `YYYYMMDD_HHMMSS_{idx}_result.json`
- **집계된 요약**: `YYYYMMDD_HHMMSS_total_result.json`

## ASCII 참조

### 맵 형식 요구 사항

- **행 일관성**: 모든 행의 길이는 동일해야 함 (`-`로 채움)
- **필수 요소**: `M` (시작) 및 `F` (종료)
- **플레이 가능성**: 레벨은 60초 이내에 완료 가능해야 함

### ASCII 타일 가이드

#### 레벨 경계
| 문자 | 설명 |
|-----------|-------------|
| `M` | 마리오 시작 위치 |
| `F` | 마리오 출구 (깃발) |
| `-` 또는 공백 | 빈 공간 (공기) |

#### 지형 및 블록
| 문자 | 설명 |
|-----------|-------------|
| `X` | 땅 (단단한 바닥) |
| `#` | 피라미드 블록 (계단/장식용) |
| `S` | 일반 벽돌 (부수기 가능) |
| `C` | 코인 벽돌 |
| `L` | 1-Up 벽돌 |
| `U` | 버섯 벽돌 |
| `D` | 사용된 블록 (이미 침) |
| `%` | 플랫폼 (점프 통과 가능) |
| `\|` | 플랫폼 배경 |

#### 물음표 블록
| 문자 | 설명 |
|-----------|-------------|
| `?` 또는 `@` | 버섯 물음표 블록 |
| `Q` 또는 `!` | 코인 물음표 블록 |
| `1` | 보이지 않는 1-Up 블록 |
| `2` | 보이지 않는 코인 블록 |

#### 아이템
| 문자 | 설명 |
|-----------|-------------|
| `o` | 수집 가능한 코인 |

#### 파이프
| 문자 | 설명 |
|-----------|-------------|
| `t` | 빈 파이프 |
| `T` | 꽃 파이프 (피라냐 식물 포함) |
| `<` `>` | 파이프 상단 (왼쪽/오른쪽) |
| `[` `]` | 파이프 몸통 (왼쪽/오른쪽) |

#### 킬러(Bullet Bill) 대포
| 문자 | 설명 |
|-----------|-------------|
| `*` | 킬러 대포 |
| `B` | 킬러(Bullet Bill) 머리 |
| `b` | 킬러 몸통 |

#### 적
| 문자 | 설명 |
|-----------|-------------|
| `E` 또는 `g` | 굼바 |
| `G` | 날개 달린 굼바 |
| `k` | 초록 엉금엉금 (Koopa) |
| `K` | 날개 달린 초록 엉금엉금 |
| `r` | 빨강 엉금엉금 |
| `R` | 날개 달린 빨강 엉금엉금 |
| `y` | 가시돌이 (Spiny) |
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

전체 ASCII 가이드는 `scenarios/mario/prompts/map_ascii_guide.md`를 참조하세요.

## 프로젝트 구조

```
agentx-vlmario/
├── scenarios/
│   └── mario/
│       ├── scenario.toml             # 벤치마크 구성
│       ├── mario_map_evaluator.py    # 마리오 맵 평가자 (Green agent)
│       ├── mario_map_designer.py     # 마리오 맵 디자이너 (Purple agent)
│       ├── PlayAstar.jar             # 게임 플레이 시뮬레이터
│       ├── prompts/                  # 프롬프트 템플릿
│       │   ├── system_prompt.md
│       │   ├── map_ascii_guide.md
│       │   ├── map_request.md
│       │   └── initial_criterion_prompt.md
│       └── img/                      # 시뮬레이션용 게임 자산
├── src/
│   └── agentbeats/                   # 핵심 프레임워크
└── .env                              # 환경설정 파일
```

## 감사의 글

- [AgentBeats](https://github.com/rdi-foundation/agentbeats-tutorial) 프레임워크 기반
- [Mario-AI-Framework](https://github.com/shyamsn97/Mario-AI-Framework) 기반의 마리오 A* 시뮬레이션
- 비전-언어 평가를 위해 Google Gemini 기반으로 구동
