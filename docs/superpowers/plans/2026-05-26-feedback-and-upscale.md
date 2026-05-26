# Feedback Chat & Preserve-Upscale Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PaperBanana 데모에 (1) candidate 결과를 이미지+텍스트로 반복 편집하는 채팅형 피드백 탭과 (2) 내용을 최대한 보존하며 고해상도화하고 변화를 시각적으로 검증하는 업스케일 탭을 추가한다.

**Architecture:** 기존 `demo.refine_image_with_nanoviz()`(Gemini image-to-image) 백엔드를 재사용한다. 참조 이미지 2장 입력을 지원하도록 helper를 확장하고, 보존 검증용 순수 함수 `compute_preservation_diff()`를 `utils/image_utils.py`에 추가한다. UI는 `st.tabs`를 2개에서 3개로 늘려 Generate / Feedback Chat / Upscale로 분리한다. 채팅 히스토리는 UX 전용이며 매 턴 모델 호출은 `현재 active 이미지 + (선택)참조 이미지 + 지시문`만 보낸다(transcript 누적 없음).

**Tech Stack:** Streamlit 1.57 (`st.chat_input(accept_file=True)`, `st.chat_message`), Pillow, numpy, google-genai (Gemini image model).

**Confirmed design defaults:**
- 되돌리기: 선택한 assistant 이미지를 `chat_active_image`로 설정, 히스토리는 보존.
- Tab3 프롬프트: 잠긴 preserve 프롬프트 + 선택적 "추가 메모" 입력란을 append.
- 탭 전환: Streamlit은 프로그램적 탭 전환을 지원하지 않으므로, candidate 액션 버튼은 `session_state`에 이미지를 적재하고 `st.toast`로 사용자에게 해당 탭 클릭을 안내한다.

---

## File Structure

- `utils/image_utils.py` — (수정) `compute_preservation_diff()` 추가. 순수 이미지 비교/오버레이 로직.
- `tests/test_preservation_diff.py` — (생성) `compute_preservation_diff`의 plain-assert 테스트(pytest 불필요, venv python으로 실행).
- `demo.py` — (수정) `refine_image_with_nanoviz` 참조이미지 확장, 탭 3개로 변경, Tab2 채팅 신규, Tab3 업스케일 리팩터, Tab1 candidate 액션 버튼, 모듈 상수/임포트 추가.
- `SETUP.md` — (수정) 실행 흐름의 탭 설명 한 줄 갱신.

---

## Task 1: 보존 검증 함수 `compute_preservation_diff`

**Files:**
- Modify: `utils/image_utils.py` (파일 끝에 함수 추가)
- Test: `tests/test_preservation_diff.py` (생성)

- [ ] **Step 1: Write the failing test**

`tests/test_preservation_diff.py` 생성:

```python
"""Plain-assert tests for compute_preservation_diff (run with venv python; no pytest needed)."""
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image
from utils.image_utils import compute_preservation_diff


def _solid(color, size=(64, 48)):
    return Image.new("RGB", size, color)


def test_identical_has_no_changes():
    img = _solid((200, 180, 160))
    out = compute_preservation_diff(img, img.copy())
    assert out["mad"] == 0.0, f"expected mad 0, got {out['mad']}"
    assert out["changed_ratio"] == 0.0, f"expected 0 changed, got {out['changed_ratio']}"
    # overlay must decode as a same-size PNG
    ov = Image.open(BytesIO(out["overlay_png_bytes"]))
    assert ov.size == img.size


def test_modified_region_is_flagged():
    base = _solid((255, 255, 255))
    mod = base.copy()
    # paint a 20x20 black square -> clearly changed pixels
    for x in range(20):
        for y in range(20):
            mod.putpixel((x, y), (0, 0, 0))
    out = compute_preservation_diff(base, mod)
    assert out["changed_ratio"] > 0.0, "modified image must report changed pixels"
    # 20x20 changed out of 64x48 = 400/3072 ~= 0.13
    assert 0.10 < out["changed_ratio"] < 0.16, f"unexpected ratio {out['changed_ratio']}"
    assert out["mad"] > 0.0


def test_upscaled_is_downscaled_to_match():
    base = _solid((120, 120, 120), size=(50, 40))
    big = base.resize((200, 160), Image.LANCZOS)  # 4x, same content
    out = compute_preservation_diff(base, big)
    assert out["changed_ratio"] == 0.0, "pure resize of a solid color must show no change"


if __name__ == "__main__":
    test_identical_has_no_changes()
    test_modified_region_is_flagged()
    test_upscaled_is_downscaled_to_match()
    print("ALL PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python tests/test_preservation_diff.py`
Expected: FAIL with `ImportError: cannot import name 'compute_preservation_diff'`

- [ ] **Step 3: Write minimal implementation**

`utils/image_utils.py` 파일 끝(line 46 이후)에 추가. 상단 import 옆에 `import numpy as np`가 필요하므로 파일 상단 import 블록(현재 `import base64`, `import io`, `from PIL import Image`)에 `import numpy as np`를 추가한다.

상단 import 수정:
```python
import base64
import io
import numpy as np
from PIL import Image
```

파일 끝에 함수 추가:
```python
def compute_preservation_diff(original, upscaled, threshold: int = 20) -> dict:
    """Compare an upscaled image against its original to visualize what changed.

    The upscaled image is downscaled back to the original size, then compared
    pixel-by-pixel. Pixels whose max per-channel difference exceeds `threshold`
    (0-255) are considered "changed" and tinted red in the overlay.

    Args:
        original: PIL.Image of the source.
        upscaled: PIL.Image of the model output (any size).
        threshold: per-pixel max-channel diff above which a pixel counts as changed.

    Returns:
        dict with:
          mad: float, mean absolute pixel difference (0-255).
          changed_ratio: float, fraction of pixels exceeding threshold (0-1).
          overlay_png_bytes: bytes, original-size PNG with changed pixels tinted red.
    """
    orig = original.convert("RGB")
    up = upscaled.convert("RGB")
    if up.size != orig.size:
        up = up.resize(orig.size, Image.LANCZOS)

    a = np.asarray(orig, dtype=np.int16)
    b = np.asarray(up, dtype=np.int16)
    abs_diff = np.abs(a - b)
    mad = float(abs_diff.mean())
    per_pixel_max = abs_diff.max(axis=2)
    mask = per_pixel_max > threshold
    changed_ratio = float(mask.mean())

    overlay = np.asarray(orig, dtype=np.uint8).copy()
    red = np.array([255, 0, 0], dtype=np.float32)
    if mask.any():
        blended = 0.5 * overlay[mask].astype(np.float32) + 0.5 * red
        overlay[mask] = blended.astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return {
        "mad": mad,
        "changed_ratio": changed_ratio,
        "overlay_png_bytes": buf.getvalue(),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python tests/test_preservation_diff.py`
Expected: `ALL PASS`

- [ ] **Step 5: Commit**

```bash
git add utils/image_utils.py tests/test_preservation_diff.py
git commit -m "feat: add compute_preservation_diff for upscale verification"
```

---

## Task 2: `refine_image_with_nanoviz` 참조 이미지(2장 입력) 지원

**Files:**
- Modify: `demo.py:168-270` (함수 시그니처 + 두 경로의 contents 구성)

- [ ] **Step 1: 시그니처와 docstring 수정**

`demo.py:168`의 함수 정의를 다음으로 교체:

```python
async def refine_image_with_nanoviz(image_bytes, edit_prompt, aspect_ratio="21:9", image_size="2K", reference_image_bytes=None):
    """
    Refine an image using an Image Editing API.
    Supports OpenRouter (priority), Google API key, and Vertex AI ADC as fallback.

    Args:
        image_bytes: Image data in bytes (the current image being edited)
        edit_prompt: Text description of desired changes
        aspect_ratio: Output aspect ratio (21:9, 16:9, 3:2)
        image_size: Output resolution (2K or 4K)
        reference_image_bytes: Optional second image (bytes) used only as a reference

    Returns:
        Tuple of (edited_image_bytes, success_message)
    """
```

- [ ] **Step 2: 참조 이미지가 있을 때 지시문에 역할 명시**

`demo.py`에서 `image_model = get_config_val(...)` 줄(현재 182) 바로 다음에 추가:

```python
    # When a reference image is attached, tell the model which image is which.
    if reference_image_bytes is not None:
        edit_prompt = (
            "You are given two images. The FIRST image is the current diagram to edit. "
            "The SECOND image is a reference provided by the user. Apply the following "
            "instruction to the first image, using the second only as a reference. "
            "Instruction: " + (edit_prompt or "Match the reference.")
        )
```

- [ ] **Step 3: OpenRouter 경로 contents에 참조 이미지 추가**

`demo.py`의 OpenRouter `contents = [...]` 블록(현재 197-200)을 교체:

```python
            contents = [
                {"type": "image", "data": image_b64, "mime_type": "image/jpeg"},
            ]
            if reference_image_bytes is not None:
                ref_b64 = base64.b64encode(reference_image_bytes).decode("utf-8")
                contents.append({"type": "image", "data": ref_b64, "mime_type": "image/jpeg"})
            contents.append({"type": "text", "text": edit_prompt})
```

- [ ] **Step 4: Gemini 경로 contents에 참조 이미지 추가**

`demo.py`의 Gemini `contents = [...]` 블록(현재 241-244)을 교체:

```python
        contents = [
            types.Part.from_text(text=edit_prompt),
            types.Part.from_bytes(mime_type="image/jpeg", data=image_bytes),
        ]
        if reference_image_bytes is not None:
            contents.append(
                types.Part.from_bytes(mime_type="image/jpeg", data=reference_image_bytes)
            )
```

- [ ] **Step 5: 구문 검증**

Run: `.venv/bin/python -c "import ast; ast.parse(open('demo.py').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 6: 실제 2-이미지 스모크 테스트 (API 1회 호출)**

기존 candidate 두 장으로 참조 편집이 이미지(에러 아님)를 반환하는지 확인. 임시 스크립트 실행:

```bash
.venv/bin/python - <<'PY'
import asyncio, sys
from io import BytesIO
from PIL import Image
sys.path.insert(0, ".")
from demo import refine_image_with_nanoviz
src = "results/auto/1._박태윤_가톨릭대_의예과_논문_A/20260520_1251/candidate_0.png"
ref = "results/auto/1._박태윤_가톨릭대_의예과_논문_A/20260520_1251/candidate_1.png"
def b(p):
    im = Image.open(p).convert("RGB"); buf = BytesIO(); im.save(buf, "JPEG", quality=95); return buf.getvalue()
out, msg = asyncio.run(refine_image_with_nanoviz(
    image_bytes=b(src), edit_prompt="Change the background tint to a very light blue.",
    aspect_ratio="16:9", image_size="2K", reference_image_bytes=b(ref)))
print("MSG:", msg, "| got image:", out is not None and len(out) > 1000)
PY
```
Expected: `MSG: ✅ Image refined successfully! ... | got image: True`

- [ ] **Step 7: Commit**

```bash
git add demo.py
git commit -m "feat: support optional reference image in refine_image_with_nanoviz"
```

---

## Task 3: 모듈 상수 + 임포트 추가 (Tab2/Tab3 준비)

**Files:**
- Modify: `demo.py` (상단 import 블록과 모듈 상수)

- [ ] **Step 1: image_utils 임포트 추가**

`demo.py`의 `from utils.pdf_ingest import extract_paper, refine_for_diagram`(현재 57) 다음 줄에 추가:

```python
    from utils.image_utils import compute_preservation_diff
```

- [ ] **Step 2: preserve 업스케일 프롬프트 상수 추가**

`demo.py`에서 `refine_image_with_nanoviz` 함수 정의 바로 앞(현재 168 직전)에 모듈 상수 추가:

```python
PRESERVE_UPSCALE_PROMPT = (
    "Upscale this scientific diagram to a higher resolution. Reproduce it EXACTLY: "
    "keep every text label, character, number, symbol, arrow, color, shape, and "
    "spatial layout identical to the input. Do NOT add, remove, rephrase, translate, "
    "or restyle anything. Only increase resolution and sharpness so the output is "
    "pixel-faithful to the input, just larger and crisper."
)
```

- [ ] **Step 3: 구문/임포트 검증**

Run: `.venv/bin/python -c "from demo import PRESERVE_UPSCALE_PROMPT; from utils.image_utils import compute_preservation_diff; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add demo.py
git commit -m "chore: add preserve-upscale prompt and image_utils import to demo"
```

---

## Task 4: 탭 구조를 3개로 확장 + 기존 Refine 탭을 Upscale 탭으로 리팩터

**Files:**
- Modify: `demo.py:417` (탭 정의), `demo.py:892-1003` (기존 Tab2 블록 → Tab3 업스케일)

- [ ] **Step 1: 탭 정의 변경**

`demo.py:417` 교체:

```python
    tab1, tab2, tab3 = st.tabs(["📊 Generate Candidates", "💬 Feedback Chat", "🔍 Upscale"])
```

- [ ] **Step 2: 기존 `with tab2:` Refine Image 블록 전체를 Tab3 업스케일로 교체**

`demo.py`의 `    # ==================== TAB 2: Refine Image ====================`부터 함수 끝의 refined-result 표시까지(현재 892-1003) 전체를 아래로 교체:

```python
    # ==================== TAB 3: Upscale (preserve) ====================
    with tab3:
        st.markdown("### 🔍 Upscale (preserve) — 내용을 바꾸지 않고 고해상도화")
        st.caption("결과를 받은 뒤, 원본과 달라진 픽셀을 빨간색으로 표시해 보존 여부를 검증합니다.")

        up_col1, up_col2, up_col3 = st.columns(3)
        with up_col1:
            up_res = st.selectbox("Resolution", ["2K", "4K"], index=1, key="up_res")
        with up_col2:
            up_ratio = st.selectbox("Aspect Ratio", ["16:9", "21:9", "3:2", "1:1"], index=0, key="up_ratio")
        with up_col3:
            up_extra = st.text_input("추가 메모 (선택)", key="up_extra",
                help="기본은 '내용 변경 없이 업스케일'입니다. 강조할 보존 포인트가 있으면 적으세요.")

        # Adopt an image sent from the Generate tab (candidate action button)
        if st.session_state.get("upscale_source_image") is not None:
            st.session_state["up_active"] = st.session_state.pop("upscale_source_image")
            st.session_state.pop("up_result", None)
            st.toast("🔍 Candidate를 업스케일 입력으로 불러왔습니다.")

        up_uploaded = st.file_uploader("또는 이미지 업로드", type=["png", "jpg", "jpeg"], key="up_uploader")
        if up_uploaded is not None:
            _img = Image.open(up_uploaded).convert("RGB")
            _buf = BytesIO(); _img.save(_buf, format="JPEG", quality=95)
            st.session_state["up_active"] = _buf.getvalue()

        if "up_active" not in st.session_state:
            st.info("Generate 탭에서 candidate의 '🔍 업스케일' 버튼을 누르거나, 위에서 이미지를 업로드하세요.")
        else:
            src_img = Image.open(BytesIO(st.session_state["up_active"])).convert("RGB")
            st.image(src_img, caption=f"Source ({src_img.size[0]}×{src_img.size[1]})", width=360)

            if st.button("🔍 Upscale (preserve)", type="primary", use_container_width=True):
                prompt = PRESERVE_UPSCALE_PROMPT
                if up_extra and up_extra.strip():
                    prompt += " Additional note: " + up_extra.strip()
                with st.spinner(f"Upscaling to {up_res} (preserving content)..."):
                    out_bytes, msg = asyncio.run(refine_image_with_nanoviz(
                        image_bytes=st.session_state["up_active"],
                        edit_prompt=prompt,
                        aspect_ratio=up_ratio,
                        image_size=up_res,
                    ))
                if out_bytes:
                    st.session_state["up_result"] = out_bytes
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

            if st.session_state.get("up_result"):
                out_img = Image.open(BytesIO(st.session_state["up_result"])).convert("RGB")
                diff = compute_preservation_diff(src_img, out_img)

                st.divider()
                st.markdown("#### 🧪 보존 검증")
                st.caption("빨간색 = 원본과 달라진 픽셀. 텍스트/라벨이 빨갛게 칠해지면 내용이 변경된 것입니다.")
                st.image(Image.open(BytesIO(diff["overlay_png_bytes"])), use_container_width=True,
                         caption=f"변화 픽셀 {diff['changed_ratio'] * 100:.2f}%")

                cmp1, cmp2 = st.columns(2)
                with cmp1:
                    st.markdown("**Before**")
                    st.image(src_img, use_container_width=True)
                with cmp2:
                    st.markdown(f"**After ({up_res}, {out_img.size[0]}×{out_img.size[1]})**")
                    st.image(out_img, use_container_width=True)

                with st.expander("정량 지표 (보조)"):
                    st.write(f"평균 절대 픽셀 차이(MAD): {diff['mad']:.2f} / 255")
                    st.write(f"변화 픽셀 비율(>20/255): {diff['changed_ratio'] * 100:.2f}%")
                    st.caption("MAD는 전체 평균이라 작은 텍스트 깨짐을 못 잡습니다. 위 빨간 오버레이가 주된 검증 신호입니다.")

                st.download_button(
                    label=f"⬇️ Download {up_res} Image",
                    data=st.session_state["up_result"],
                    file_name=f"upscaled_{up_res}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True,
                )
```

- [ ] **Step 3: 구문 검증**

Run: `.venv/bin/python -c "import ast; ast.parse(open('demo.py').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 4: Commit**

```bash
git add demo.py
git commit -m "feat: refactor Refine tab into preserve-Upscale tab with diff verification"
```

---

## Task 5: Tab2 — Feedback Chat (신규)

**Files:**
- Modify: `demo.py` (Task 4에서 만든 `with tab3:` 블록 **앞**에 `with tab2:` 블록 삽입)

- [ ] **Step 1: 채팅 탭 블록 삽입**

`demo.py`에서 `    # ==================== TAB 3: Upscale (preserve) ====================` 줄 **바로 앞**에 아래 블록을 삽입:

```python
    # ==================== TAB 2: Feedback Chat ====================
    with tab2:
        st.markdown("### 💬 Feedback Chat — 텍스트 또는 이미지+텍스트로 반복 편집")
        st.caption("Generate 탭에서 candidate를 보내거나 이미지를 업로드한 뒤, 채팅처럼 수정 지시를 주세요.")

        # Adopt an image sent from the Generate tab (resets the conversation)
        if st.session_state.get("chat_base_image") is not None:
            st.session_state["chat_active_image"] = st.session_state.pop("chat_base_image")
            st.session_state["chat_history"] = []
            st.toast("💬 Candidate를 Feedback Chat으로 불러왔습니다.")

        chat_uploaded = st.file_uploader("또는 이미지 업로드로 시작", type=["png", "jpg", "jpeg"], key="chat_uploader")
        if chat_uploaded is not None and st.button("업로드한 이미지로 시작", key="chat_use_upload"):
            _img = Image.open(chat_uploaded).convert("RGB")
            _buf = BytesIO(); _img.save(_buf, format="JPEG", quality=95)
            st.session_state["chat_active_image"] = _buf.getvalue()
            st.session_state["chat_history"] = []
            st.rerun()

        if "chat_active_image" not in st.session_state:
            st.info("Generate 탭에서 candidate의 '💬 피드백' 버튼을 누르거나, 위에서 이미지를 업로드하세요.")
        else:
            st.markdown("**현재 이미지 (편집은 이 이미지에 적용됩니다):**")
            st.image(Image.open(BytesIO(st.session_state["chat_active_image"])), width=320)

            st.session_state.setdefault("chat_history", [])
            for i, turn in enumerate(st.session_state["chat_history"]):
                with st.chat_message(turn["role"]):
                    if turn["role"] == "user":
                        if turn.get("text"):
                            st.write(turn["text"])
                        if turn.get("ref_image"):
                            st.image(Image.open(BytesIO(turn["ref_image"])), width=160, caption="첨부 참조 이미지")
                    else:
                        if turn.get("image"):
                            st.image(Image.open(BytesIO(turn["image"])), use_container_width=True)
                            if st.button("↩️ 이 버전으로 되돌리기", key=f"revert_{i}"):
                                st.session_state["chat_active_image"] = turn["image"]
                                st.toast("되돌렸습니다. 다음 편집은 이 버전에 적용됩니다.")
                                st.rerun()
                        if turn.get("text"):
                            st.caption(turn["text"])

            user_msg = st.chat_input(
                "수정할 내용을 입력하세요 (이미지 첨부 가능)",
                accept_file=True,
                file_type=["png", "jpg", "jpeg"],
            )
            if user_msg:
                instruction = getattr(user_msg, "text", "") or ""
                files = getattr(user_msg, "files", None) or []
                ref_bytes = None
                if files:
                    _ref = Image.open(files[0]).convert("RGB")
                    _rbuf = BytesIO(); _ref.save(_rbuf, format="JPEG", quality=95)
                    ref_bytes = _rbuf.getvalue()

                if not instruction.strip() and ref_bytes is None:
                    st.warning("수정 지시를 입력하세요.")
                else:
                    st.session_state["chat_history"].append(
                        {"role": "user", "text": instruction, "ref_image": ref_bytes, "image": None}
                    )
                    with st.spinner("편집 중..."):
                        edited, msg = asyncio.run(refine_image_with_nanoviz(
                            image_bytes=st.session_state["chat_active_image"],
                            edit_prompt=instruction or "Apply the change shown in the reference image.",
                            aspect_ratio=st.session_state.get("chat_aspect_ratio", "16:9"),
                            image_size="2K",
                            reference_image_bytes=ref_bytes,
                        ))
                    if edited:
                        st.session_state["chat_active_image"] = edited
                        st.session_state["chat_history"].append(
                            {"role": "assistant", "text": msg, "ref_image": None, "image": edited}
                        )
                    else:
                        st.session_state["chat_history"].append(
                            {"role": "assistant", "text": msg, "ref_image": None, "image": None}
                        )
                    st.rerun()
```

- [ ] **Step 2: 구문 검증**

Run: `.venv/bin/python -c "import ast; ast.parse(open('demo.py').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add demo.py
git commit -m "feat: add Feedback Chat tab for iterative image+text editing"
```

---

## Task 6: Tab1 candidate 카드에 액션 버튼 추가

**Files:**
- Modify: `demo.py:347-366` (`display_candidate_result`의 다운로드 버튼 블록)

- [ ] **Step 1: 다운로드 버튼 다음에 두 액션 버튼 추가**

`demo.py`의 `display_candidate_result` 안, 다운로드 `st.download_button(...)` 호출이 끝나는 지점(현재 362, `)` 다음, `else:`(363) 전)에 아래를 삽입:

```python
            # Action buttons: send this candidate to the Feedback Chat or Upscale tab
            jpg_bytes = base64.b64decode(result[final_image_key])
            act1, act2 = st.columns(2)
            with act1:
                if st.button("💬 피드백", key=f"feedback_candidate_{candidate_id}", use_container_width=True):
                    st.session_state["chat_base_image"] = jpg_bytes
                    st.toast("💬 'Feedback Chat' 탭으로 보냈습니다. 상단 탭을 클릭하세요.")
            with act2:
                if st.button("🔍 업스케일", key=f"upscale_candidate_{candidate_id}", use_container_width=True):
                    st.session_state["upscale_source_image"] = jpg_bytes
                    st.toast("🔍 'Upscale' 탭으로 보냈습니다. 상단 탭을 클릭하세요.")
```

> 참고: `result[final_image_key]`는 base64(jpg) 문자열이며 `base64.b64decode`로 곧장 JPEG bytes가 된다. 이 블록은 `if img:`(이미지 디코딩 성공) 분기 안에 있어 `final_image_key`가 항상 유효하다.

- [ ] **Step 2: 구문 검증**

Run: `.venv/bin/python -c "import ast; ast.parse(open('demo.py').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add demo.py
git commit -m "feat: add per-candidate Feedback/Upscale action buttons"
```

---

## Task 7: SETUP.md 갱신 + 최종 스모크 검증

**Files:**
- Modify: `SETUP.md` (옵션 A 실행 흐름 섹션)

- [ ] **Step 1: SETUP.md 탭 설명 갱신**

`SETUP.md`의 "옵션 A — Streamlit UI" 사용 흐름 목록(현재 6단계) 다음에 한 줄 추가:

```markdown
7. 생성된 candidate 카드의 **💬 피드백** / **🔍 업스케일** 버튼으로 각각 *Feedback Chat* 탭(반복 편집)·*Upscale* 탭(보존 업스케일+검증)으로 보낼 수 있음
```

- [ ] **Step 2: 전체 import 스모크 검증**

Run:
```bash
.venv/bin/python -c "import demo; print('demo import OK')"
.venv/bin/python tests/test_preservation_diff.py
```
Expected: `demo import OK` 그리고 `ALL PASS`

- [ ] **Step 3: Streamlit 기동 스모크 (5초)**

Run:
```bash
.venv/bin/streamlit run demo.py --server.headless true --server.port 8599 &
sleep 5 && curl -s -o /dev/null -w "%{http_code}" http://localhost:8599 && kill %1
```
Expected: `200`

- [ ] **Step 4: Commit**

```bash
git add SETUP.md
git commit -m "docs: document Feedback Chat and Upscale tabs in SETUP.md"
```

---

## Self-Review (완료)

**Spec coverage:**
- 기능1(피드백 채팅, 이미지+텍스트, 반복, LLM 솔루션형 UX) → Task 5 (+ Task 2 참조이미지, Task 6 진입버튼). ✓
- 기능2(보존 업스케일 + 검증) → Task 4 (+ Task 1 검증함수, Task 3 프롬프트). ✓
- Tab1 candidate 진입점 → Task 6. ✓
- helper 2장 입력 → Task 2. ✓
- 검증 시각 오버레이 메인 / MAD 보조 → Task 1 + Task 4. ✓

**Placeholder scan:** 모든 step에 실제 코드/명령 포함, TBD 없음. ✓

**Type consistency:** `compute_preservation_diff`가 반환하는 키(`mad`, `changed_ratio`, `overlay_png_bytes`)는 Task 1 정의와 Task 4 사용처가 일치. `refine_image_with_nanoviz`의 `reference_image_bytes` 파라미터는 Task 2 정의와 Task 5 사용처 일치. session_state 키(`chat_base_image`, `chat_active_image`, `chat_history`, `upscale_source_image`, `up_active`, `up_result`)는 생산/소비 지점이 일치. ✓
