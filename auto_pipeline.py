"""End-to-end automation: PDF -> PaperBanana diagrams.

Usage:
    python auto_pipeline.py path/to/paper.pdf
    python auto_pipeline.py path/to/paper.pdf --figure 2 --candidates 4
    python auto_pipeline.py path/to/paper.pdf --extract-only   # skip generation

Pipeline:
    1. pymupdf  -> raw text + section split + figure legends   (utils.pdf_ingest)
    2. Gemini   -> structured method markdown + caption        (utils.pdf_ingest.refine_for_diagram)
    3. PaperBanana demo_full pipeline                          (demo.process_parallel_candidates)
    4. Save PNGs + metadata under results/auto/<stem>/<ts>/
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from demo import create_sample_inputs, process_parallel_candidates
from utils.pdf_ingest import (
    Paper,
    RefinedInput,
    extract_paper,
    refine_for_diagram,
    refined_to_dict,
)


def _save_extraction(out_dir: Path, paper: Paper, refined: RefinedInput) -> None:
    (out_dir / "extracted_method.md").write_text(refined.method_markdown, encoding="utf-8")
    (out_dir / "extracted_caption.md").write_text(refined.caption, encoding="utf-8")
    (out_dir / "extraction_meta.json").write_text(
        json.dumps(
            {
                "paper_title": paper.title,
                "sections_found": list(paper.sections.keys()),
                "available_figures": paper.available_figures(),
                "target_figure": refined.target_figure,
                "diagram_type": refined.diagram_type,
                "aspect_ratio": refined.aspect_ratio,
                "key_entities": refined.key_entities,
                "key_relations": refined.key_relations,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _extract_final_image_key(result: dict, task: str = "diagram") -> str | None:
    for r in range(3, -1, -1):
        k = f"target_{task}_critic_desc{r}_base64_jpg"
        if k in result and result[k]:
            return k
    for k in (f"target_{task}_stylist_desc0_base64_jpg", f"target_{task}_desc0_base64_jpg"):
        if k in result and result[k]:
            return k
    return None


def _save_candidates(out_dir: Path, results: list[dict]) -> list[Path]:
    saved = []
    for i, result in enumerate(results):
        key = _extract_final_image_key(result)
        if not key:
            print(f"  candidate {i}: no image found")
            continue
        png_bytes = base64.b64decode(result[key])
        png_path = out_dir / f"candidate_{i}.png"
        png_path.write_bytes(png_bytes)
        saved.append(png_path)
        print(f"  candidate {i}: saved {png_path.name} ({len(png_bytes)//1024} KB)")

    truncated = []
    for r in results:
        truncated.append({
            k: (v if not (isinstance(v, str) and len(v) > 300) else v[:300] + "...<truncated>")
            for k, v in r.items()
        })
    (out_dir / "raw_results.json").write_text(
        json.dumps(truncated, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return saved


async def run(args: argparse.Namespace) -> None:
    pdf_path = Path(args.pdf).expanduser().resolve()
    stem = pdf_path.stem.replace(" ", "_")[:60]
    ts = time.strftime("%Y%m%d_%H%M")
    out_dir = Path(__file__).parent / "results" / "auto" / stem / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    print("\n[1/3] Extracting text and sections from PDF...")
    paper = extract_paper(pdf_path)
    print(f"  title: {paper.title}")
    print(f"  sections: {list(paper.sections.keys())}")
    print(f"  figures: {paper.available_figures()}")
    print(f"  methods: {len(paper.methods)} chars")

    if not paper.methods:
        print("  WARNING: Methods section not detected. The LLM will fall back to full text.")

    print("\n[2/3] Refining for diagram via Gemini...")
    refined = await refine_for_diagram(paper, figure_num=args.figure)
    print(f"  target figure: {refined.target_figure}")
    print(f"  diagram_type: {refined.diagram_type}, aspect_ratio: {refined.aspect_ratio}")
    print(f"  caption: {refined.caption[:120]}...")
    print(f"  key_entities: {refined.key_entities[:6]}")
    _save_extraction(out_dir, paper, refined)
    print(f"  → saved extracted_method.md, extracted_caption.md, extraction_meta.json")

    if args.extract_only:
        print("\n--extract-only set; skipping generation.")
        return

    print(f"\n[3/3] Running PaperBanana ({args.exp_mode}, {args.candidates} candidates)...")
    inputs = create_sample_inputs(
        method_content=refined.method_markdown,
        caption=refined.caption,
        diagram_type=refined.diagram_type,
        aspect_ratio=refined.aspect_ratio,
        num_copies=args.candidates,
        max_critic_rounds=args.critic_rounds,
    )
    results = await process_parallel_candidates(
        inputs,
        exp_mode=args.exp_mode,
        retrieval_setting=args.retrieval,
    )
    print(f"  got {len(results)} results, extracting images...")
    saved = _save_candidates(out_dir, results)

    print(f"\n=== Done. {len(saved)} PNG(s) in {out_dir} ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF → PaperBanana diagram automation.")
    parser.add_argument("pdf", help="Path to the source PDF.")
    parser.add_argument("--figure", type=int, default=None,
                        help="Target figure number. Default: auto-pick overview/schematic figure.")
    parser.add_argument("--candidates", type=int, default=4,
                        help="Number of parallel diagram candidates (default 4).")
    parser.add_argument("--critic-rounds", type=int, default=2,
                        help="Max critic refinement rounds (default 2).")
    parser.add_argument("--exp-mode", default="demo_full",
                        choices=["demo_full", "demo_planner_critic"],
                        help="PaperBanana pipeline mode (default demo_full).")
    parser.add_argument("--retrieval", default="auto",
                        choices=["auto", "manual", "random", "none"],
                        help="Retriever setting for reference diagrams (default auto).")
    parser.add_argument("--extract-only", action="store_true",
                        help="Run text extraction + LLM refine only; skip image generation.")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
