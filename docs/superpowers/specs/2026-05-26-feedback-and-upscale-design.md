# Feedback Chat & Preserve-Upscale — Design

**Date:** 2026-05-26
**Status:** Approved (structure), pending spec review
**Scope:** `demo.py` (Streamlit UI) + small helper extensions in `utils/`

## 1. Goals

PaperBanana가 생성한 다이어그램 결과물(candidate)에 대해 사용자가:

1. **(기능1) 피드백 기반 반복 편집** — 텍스트 또는 이미지+텍스트로 추가 지시를 주면
   이미지를 수정. 일반적인 LLM 채팅 솔루션 같은 UX (대화형, 히스토리, 되돌리기).
2. **(기능2) 보존 업스케일** — 결과 이미지를 **최대한 변경하지 않고** 고해상도화. 그리고
   실제로 안 바뀌었는지 **검증**할 수 있어야 함.

두 기능은 의도가 정반대다: 기능1은 "바꿔라", 기능2는 "바꾸지 마라". 따라서 UI에서
명확히 분리한다.

## 2. Key technical findings (evidence)

기능2의 핵심 우려는 "재생성 계열 이미지 모델이 다이어그램의 텍스트 라벨을 깨뜨린다"는
것이었다. 기존 `demo.refine_image_with_nanoviz()`(Gemini 이미지 모델 image-to-image) 경로로
실제 candidate(`results/auto/1._박태윤.../20260520_1251/candidate_0.png`, 1376×768, 한글·그리스·
작은 영문 라벨 다수)를 강한 "preserve everything" 프롬프트로 4K 업스케일해 검증했다:

| 항목 | 결과 |
|---|---|
| 해상도 | 1376×768 → 5504×3072 (4×) |
| 평균 절대 픽셀 차이 (다운스케일 후) | 4.41 / 255 (~1.7%) |
| 한글 텍스트("급성 신경염증" 등) | 완전 보존 |
| 그리스 문자/화살표 (α/β/γ, ↑) | 보존 |
| 작은 라벨 (Foxp3, Needle trauma) | 보존 |
| 레이아웃/색상 | 동일, 선명도만 향상 |

**결론:** 이 특정 Gemini 이미지 모델 + 강한 preserve 프롬프트는 보존 업스케일에 실용적으로
충분하다. 따라서 기능2는 별도 SR 모델(Real-ESRGAN, torch 의존성) 도입 없이 기존 백엔드를
재사용한다. 단, 단일 샘플이고 모델 변동성이 있으므로 **검증을 기능에 내장**한다.

> probe 스크립트와 산출물은 일회성(`/tmp/upscale_probe*.py/png`)으로 커밋하지 않는다.

## 3. Plumbing 변경 (공유 백엔드)

### 3.1 `refine_image_with_nanoviz` 확장 — 참조 이미지(2장 입력) 지원

기능1에서 사용자가 참조 이미지를 첨부하면, 모델에 **active 이미지 + reference 이미지** 2장을
보내야 한다. 현재 함수는 단일 `image_bytes`만 받는다.

- 시그니처에 `reference_image_bytes: bytes | None = None` 추가.
- reference가 있으면 contents는 `[현재 이미지 part, 참조 이미지 part, 텍스트 part]`.
- 프롬프트에 역할을 명시: *"The first image is the current diagram you are editing.
  The second image is a reference provided by the user."* — 모델이 두 이미지를 혼동하지 않도록.
- OpenRouter / Google API key / Vertex 세 경로 모두에 동일하게 반영.

이 변경은 기존 단일-이미지 호출(reference=None)과 하위 호환된다.

### 3.2 검증 유틸 — `utils/image_utils.py`에 추가

`compute_preservation_diff(original: Image, upscaled: Image) -> dict`:

- upscaled를 original 크기로 Lanczos 다운스케일.
- 픽셀 절대 차이 계산 (numpy, 이미 의존성에 있음 — scikit-image 추가 안 함).
- 반환: `{"mad": float, "changed_mask_png_bytes": bytes, "overlay_png_bytes": bytes}`
  - `overlay`: original 위에 차이>임계치(기본 20/255) 픽셀을 반투명 빨강으로 칠한 이미지.
    **이것이 메인 검증 신호** — 텍스트가 깨지면 글자 영역이 빨갛게 보인다.
  - `mad`는 보조 숫자(전체 픽셀 평균이라 텍스트 깨짐을 못 잡으므로 헤드라인으로 쓰지 않음).

## 4. UI 구조 (3 탭)

```
Tab1: 📊 Generate Candidates   (기존 + candidate 카드에 액션 버튼)
Tab2: 💬 Feedback Chat          (신규)
Tab3: 🔍 Upscale (preserve)     (기존 Refine Image 탭을 업스케일 전용으로 리팩터)
```

### 4.1 Tab1 — candidate 카드 액션 버튼

`display_candidate_result()`의 다운로드 버튼 옆에 두 버튼 추가:

- **💬 이 결과로 피드백** → 해당 candidate의 최종 이미지 base64를
  `st.session_state["chat_base_image"]`에 싣고, 채팅 히스토리를 초기화한 뒤
  Tab2로 안내(Streamlit은 프로그램적 탭 전환이 제한적이므로 `st.toast`/안내문 + 세션 적재).
- **🔍 업스케일** → 동일 이미지를 `st.session_state["upscale_source_image"]`에 싣고 Tab3로 안내.

candidate 이미지는 base64(jpg)로 이미 result dict에 있으므로 그대로 옮긴다.

### 4.2 Tab2 — Feedback Chat (기능1)

**진입:** (a) Tab1에서 "피드백" 버튼으로 적재된 `chat_base_image`, 또는 (b) 상단
file_uploader로 직접 업로드. 둘 중 하나로 **active image**가 정해진다.

**상태 (session_state):**
- `chat_active_image: bytes` — 다음 편집의 베이스가 되는 현재 이미지.
- `chat_history: list[dict]` — 각 항목:
  `{"role": "user"|"assistant", "text": str, "ref_image": bytes|None, "image": bytes|None}`.

**UX:**
- `st.chat_message`로 히스토리 렌더 (user 말풍선엔 지시문+첨부 썸네일, assistant 말풍선엔
  결과 이미지 + "↩️ 이 버전으로 되돌리기" 버튼).
- `st.chat_input(accept_file=True, file_type=["png","jpg","jpeg"])` (Streamlit 1.57 지원 확인됨)
  으로 텍스트+선택적 이미지 첨부를 한 입력에서 받는다.

**한 턴 처리 (중요 — 모델 호출 의미):**
- 모델에 보내는 것은 **`chat_active_image` + (있으면) 첨부 reference + 지시문 텍스트**뿐.
  전체 대화 transcript를 모델에 먹이지 **않는다**. 이미지 편집 모델은 단일-턴이며, 채팅
  히스토리는 순수 UX/탐색용이다.
- 결과 이미지를 받으면 history에 user/assistant 두 항목 append, 그리고 결과를
  `chat_active_image`로 갱신(=다음 편집은 최신 결과 위에서).
- "되돌리기"는 선택한 assistant 이미지를 `chat_active_image`로 되돌린다(히스토리는 보존).

**해상도:** 편집은 다이어그램 기본 비율(예: 16:9)에서 동작. 업스케일과 달리 여기선 "변경"이
목적이므로 preserve 프롬프트를 강제하지 않는다.

### 4.3 Tab3 — Upscale (기능2, 기존 탭 리팩터)

- 진입: Tab1에서 적재된 `upscale_source_image` 또는 file_uploader 업로드.
- 사용자 편집 프롬프트 입력란 제거(또는 "추가 메모(선택)"로 축소). 기본은 **잠긴 preserve
  프롬프트**(probe에서 검증된 문구)를 사용.
- 설정: 해상도 2K/4K, 비율(원본에서 자동 감지 + 수동 override).
- 결과 표시:
  1. **변화 오버레이(메인)** — `compute_preservation_diff`의 overlay. "빨간 영역 = 바뀐 곳".
  2. **side-by-side** — 원본 vs 업스케일.
  3. expander 안에 보조 숫자(MAD)와 다운로드 버튼.

## 5. Error handling

- 모델이 이미지 대신 텍스트/에러 반환 → 기존 `refine_image_with_nanoviz`의 `(None, msg)`
  패턴 유지, UI에 에러 표시 후 active image 불변.
- 업스케일 결과 비율이 원본과 다르면(모델이 비율 무시) overlay 계산은 다운스케일 매칭으로
  흡수. 큰 비율 변화 시 경고 표시.
- API 키 없음 → 기존 RuntimeError 메시지 노출.
- 빈 입력(텍스트도 첨부도 없음) → 호출 안 함, 안내.

## 6. Testing

- 단위: `compute_preservation_diff`를 동일 이미지(diff=0)와 텍스트 1글자만 바꾼 이미지로
  테스트 → 동일은 overlay 빈칸, 변경은 해당 영역 빨강 확인.
- 단위: `refine_image_with_nanoviz(reference_image_bytes=...)`가 2-part contents를 구성하는지
  (네트워크는 mock).
- 수동/스모크: 실제 candidate로 (a) 피드백 1~2턴 편집, (b) 보존 업스케일 후 overlay가
  텍스트 영역을 빨갛게 칠하지 않는지 육안 확인.

## 7. Out of scope (YAGNI)

- Real-ESRGAN/torch 기반 SR (검증 결과 불필요).
- 대화 transcript를 모델 context로 누적 (단일-턴 모델이라 무의미).
- 편집 히스토리 영속화(디스크 저장) — 세션 내 메모리로 충분. 다운로드 버튼으로 보존 가능.
- 멀티 candidate 동시 편집/배치.

## 8. Files touched

- `demo.py` — Tab2 신규, Tab3 리팩터, Tab1 candidate 버튼, `refine_image_with_nanoviz` 확장.
- `utils/image_utils.py` — `compute_preservation_diff` 추가.
- `SETUP.md` — 새 탭 사용법 한 줄 갱신(의존성 변화 없음).
