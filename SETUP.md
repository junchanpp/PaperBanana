# PaperBanana Fork — 세팅 가이드

이 문서를 **Claude Code (또는 다른 AI 에이전트)** 에게 그대로 던지면 한 번에 세팅이 끝나도록 작성됨.

> **For Claude Code:** 아래 단계를 처음부터 끝까지 순서대로 수행해줘.
> 각 명령은 working directory 유지한 채 진행. 사용자에게 물어볼 건 단 하나, **Google Gemini API key** (`AIza...` 형식). 다른 옵션은 그냥 기본값으로 가.

---

## 0. Prerequisites

| 도구 | 설치 확인 명령 | 설치 방법 |
|---|---|---|
| `uv` ≥ 0.4 | `uv --version` | <https://docs.astral.sh/uv/getting-started/installation/> |
| `git` | `git --version` | (대부분 설치되어 있음) |
| (선택) `gh` CLI | `gh auth status` | <https://cli.github.com> |
| **Google Gemini API key** | — | <https://aistudio.google.com/apikey> 에서 발급 (무료) |

OS: macOS / Linux. Windows는 WSL 권장.

---

## 1. Clone

```bash
git clone https://github.com/junchanpp/PaperBanana.git
cd PaperBanana
```

> 본 fork에는 **PDF 자동 추출 파이프라인**이 추가되어 있음 (`utils/pdf_ingest.py`, `auto_pipeline.py`, `demo.py`의 "Auto-fill from PDF" UI). 원본 `dwzhu-pku/PaperBanana`와 비교하려면 `git diff upstream/main` (upstream remote는 직접 추가).

---

## 2. Python 3.12 가상환경 생성

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
```

활성화 확인: `which python` → `.../PaperBanana/.venv/bin/python`이 나와야 함.

---

## 3. 의존성 설치

```bash
uv pip install -r requirements.txt
```

google-genai, streamlit, pymupdf 등 약 70개 패키지 설치 (1~2분). 끝나면:

```bash
python -c "import fitz, streamlit, google.genai; print('imports OK')"
```

---

## 4. Gemini API key 설정

template을 복사한 뒤 key를 채워넣음:

```bash
cp configs/model_config.template.yaml configs/model_config.yaml
```

`configs/model_config.yaml` 파일을 열어 `api_keys.google_api_key` 줄을 다음처럼 수정:

```yaml
api_keys:
  google_api_key: "AIza..."   # ← 여기에 사용자의 키를 넣음
  openai_api_key: ""
  anthropic_api_key: ""
  openrouter_api_key: ""
```

이 파일은 `.gitignore`에 등록되어 있어 커밋되지 않음.

기본 모델 설정은 그대로 두면 됨:
- `defaults.main_model_name: "gemini-3.1-pro-preview"`
- `defaults.image_gen_model_name: "gemini-3.1-flash-image-preview"`

**검증:**
```bash
python -c "from utils.config import ExpConfig; c = ExpConfig(dataset_name='PaperBananaBench'); print('main:', c.main_model_name, '/ image:', c.image_gen_model_name)"
```

---

## 5. (선택, 강력 권장) Reference dataset 다운로드 — 약 254 MB

PaperBananaBench는 610개의 학술 다이어그램 reference 모음. Retriever Agent가 이 중에서 입력에 어울리는 10개를 골라 Planner/Stylist에게 in-context 예시로 제공.

> 다운로드 안 해도 `--retrieval none` 으로 작동하긴 함. 다만 결과 품질이 눈에 띄게 떨어짐.

```bash
mkdir -p data
hf download dwzhu/PaperBananaBench --repo-type dataset --local-dir data/PaperBananaBench
cd data/PaperBananaBench
unzip -q PaperBananaBench.zip
mv PaperBananaBench/* .
rmdir PaperBananaBench
rm PaperBananaBench.zip
cd ../..
```

**검증:**
```bash
ls data/PaperBananaBench/diagram/images/ | wc -l   # 610 이 나와야 함
```

> `huggingface-cli`는 deprecated. 반드시 `hf` 명령을 사용.

---

## 6. 실행

### 옵션 A — Streamlit UI (권장)

```bash
streamlit run demo.py
```

브라우저에 자동으로 <http://localhost:8501> 열림. 사용 흐름:

1. **Generate Candidates** 탭 선택
2. **📄 Auto-fill from PDF (optional)** expander 펼침
3. PDF 업로드
4. Refinement mode 선택:
   - **Smart refine** (권장) — LLM이 finding 중심 markdown으로 압축 (~10초)
   - **Raw dump** — 전체 PDF 텍스트를 그대로 입력 (즉시, fallback용)
5. **✨ Extract & Fill** 클릭 → Method/Caption 자동 채워짐
6. 검토 후 **🚀 Generate Candidates** 클릭 → 1~5분 후 다이어그램 후보들 표시

### 옵션 B — CLI

```bash
python auto_pipeline.py path/to/paper.pdf --candidates 4
```

옵션:
- `--candidates N` (기본 4): 병렬 생성 후보 수
- `--critic-rounds N` (기본 2): Critic agent 반복 횟수
- `--exp-mode demo_full|demo_planner_critic` (기본 `demo_full`)
- `--retrieval auto|manual|random|none` (기본 `auto`)
- `--extract-only`: LLM 정제 결과만 보고 이미지 생성은 스킵 (빠른 검증용)

결과는 `results/auto/<paper_stem>/<YYYYMMDD_HHMM>/` 아래에:
- `extracted_method.md`, `extracted_caption.md`, `extraction_meta.json`
- `candidate_0.png` ~ `candidate_{N-1}.png`
- `raw_results.json` (Planner/Stylist/Critic 단계별 텍스트)

---

## 7. End-to-end Smoke test (Claude Code가 자동 수행)

```bash
source .venv/bin/activate
python -c "from utils.pdf_ingest import extract_paper; print('pdf_ingest import OK')"
python -c "from utils.config import ExpConfig; ExpConfig(dataset_name='PaperBananaBench')"
python -c "from demo import create_sample_inputs; print('demo import OK')"
```

세 줄 모두 에러 없이 출력되면 세팅 완료.

추가로 진짜 PDF가 있다면 빠른 검증:
```bash
python auto_pipeline.py <paper.pdf> --extract-only
```
약 15초 후 `results/auto/.../extracted_method.md`가 생성되고 안에 markdown이 있으면 성공.

---

## Troubleshooting

| 증상 | 원인 / 해결 |
|---|---|
| `huggingface-cli: command not found` | `hf` 명령으로 대체 (`huggingface_hub` 0.26+에서 명칭 변경) |
| `pymupdf` 설치 실패 | `uv pip install --upgrade pip wheel` 후 재시도. Apple Silicon은 보통 wheel이 있어 문제 없음 |
| Streamlit이 안 열림 | `streamlit run demo.py --server.headless true --server.port 8501` 후 브라우저 수동 열기 |
| `RuntimeError: Gemini client was not initialized` | `configs/model_config.yaml`의 `google_api_key`가 비어있거나 잘못됨. 따옴표 안 닫혔는지 확인 |
| PDF Methods 추출이 0자 | 스캔본 PDF (텍스트 레이어 없음). OCR은 미지원 |
| `Methods section not found` 경고 | 정규식이 헤딩을 못 잡음. 정상 동작은 함 (LLM이 전체 텍스트에서 추론). 무시 가능 |
| Critic 라운드 도중 빈 응답 | Gemini API의 일시적 문제. 자동 retry됨. 5번 연속 실패하면 API key/모델 확인 |

---

## 디렉토리 구조 (이 fork에서 추가/수정된 것)

```
PaperBanana/
├── auto_pipeline.py           # NEW — CLI 진입점
├── utils/
│   └── pdf_ingest.py          # NEW — pymupdf 추출 + Gemini 정제
├── demo.py                    # MODIFIED — "Auto-fill from PDF" UI 통합
├── requirements.txt           # MODIFIED — pymupdf 추가
└── SETUP.md                   # NEW — 이 문서
```

원본 PaperBanana의 5-agent 파이프라인 (Retriever/Planner/Stylist/Visualizer/Critic)은 손대지 않음.

---

## Notes for AI agents

- 이 fork의 origin: `https://github.com/junchanpp/PaperBanana.git`
- 원본 (upstream): `https://github.com/dwzhu-pku/PaperBanana.git`
- 추가 기능 작업 시 새 branch 만들고 `gh pr create --base junchanpp:main` 으로 fork에 PR.
- API key는 `configs/model_config.yaml`에만 저장. 절대 commit하지 말 것 (`.gitignore`로 보호되긴 하지만 stage 전 한 번 더 확인).
- Streamlit 디버깅 시 백그라운드 실행 후 `tail -f` 로 로그 확인.
- PDF 자동 추출이 헤딩을 못 잡는 경우 `utils/pdf_ingest.py:SECTION_HEADINGS` 리스트에 키워드 추가.
- LLM 프롬프트 튜닝은 `utils/pdf_ingest.py:REFINE_SYSTEM_PROMPT` 한 곳에서만 함.
