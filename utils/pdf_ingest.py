"""PDF ingestion: extract paper text, split sections, and refine for PaperBanana input.

Two main entry points:
- extract_paper(pdf_path) -> Paper: pure text extraction + heuristic section split.
- refine_for_diagram(paper, figure_num=None) -> RefinedInput: LLM call that returns
  structured method markdown + caption tuned for PaperBanana's downstream agents.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import fitz  # pymupdf
import json_repair
from google.genai import types

from utils.config import ExpConfig
from utils.generation_utils import call_gemini_with_retry_async


SECTION_HEADINGS = [
    "abstract",
    "introduction",
    "background",
    "related work",
    "materials and methods",
    "methods",
    "experimental",
    "results",
    "discussion",
    "conclusion",
    "references",
    "acknowledgements",
    "acknowledgments",
    "supplementary",
    "extended data",
]

FIGURE_LEGEND_RE = re.compile(
    r"(?im)^\s*(?:extended\s+data\s+)?(?:figure|fig\.?)\s*(\d+)\s*[\.\:\|]\s*(.{20,})"
)


@dataclass
class Paper:
    """Plain extraction result. No LLM involved."""
    title: str
    full_text: str
    sections: dict = field(default_factory=dict)  # name -> text
    figure_legends: dict = field(default_factory=dict)  # int -> caption text

    @property
    def methods(self) -> str:
        for key in ("materials and methods", "methods", "experimental"):
            if key in self.sections:
                return self.sections[key]
        return ""

    @property
    def results(self) -> str:
        return self.sections.get("results", "")

    @property
    def discussion(self) -> str:
        for key in ("discussion", "conclusion"):
            if key in self.sections:
                return self.sections[key]
        return ""

    @property
    def abstract(self) -> str:
        return self.sections.get("abstract", "")

    def available_figures(self) -> list[int]:
        return sorted(self.figure_legends.keys())


@dataclass
class RefinedInput:
    """LLM-refined input ready to feed into PaperBanana's create_sample_inputs."""
    method_markdown: str
    caption: str
    diagram_type: str = "Pipeline"
    aspect_ratio: str = "16:9"
    key_entities: list = field(default_factory=list)
    key_relations: list = field(default_factory=list)
    target_figure: int | None = None


def _strip_repeating_lines(pages: list[str]) -> list[str]:
    """Remove headers/footers that repeat across many pages."""
    if len(pages) < 4:
        return pages
    line_counts: Counter = Counter()
    for p in pages:
        for line in p.splitlines():
            s = line.strip()
            if 0 < len(s) <= 120:
                line_counts[s] += 1
    threshold = max(3, len(pages) // 3)
    junk = {line for line, c in line_counts.items() if c >= threshold}
    cleaned = []
    for p in pages:
        kept = [line for line in p.splitlines() if line.strip() not in junk]
        cleaned.append("\n".join(kept))
    return cleaned


def _extract_title(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if 10 < len(s) < 200 and not s.lower().startswith(("author", "page", "doi", "http")):
            return s
    return ""


def _split_sections(text: str) -> dict[str, str]:
    """Find known section headings (case-insensitive, anchored at line start) and slice text."""
    pattern = re.compile(
        r"(?im)^\s*(" + "|".join(re.escape(h) for h in SECTION_HEADINGS) + r")\b\s*$",
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return {}
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if name not in sections or len(body) > len(sections[name]):
            sections[name] = body
    return sections


def _extract_figure_legends(text: str) -> dict[int, str]:
    """Pull each 'Figure N.' caption block. Captions end at next blank line or next figure heading."""
    legends: dict[int, str] = {}
    matches = list(FIGURE_LEGEND_RE.finditer(text))
    for i, m in enumerate(matches):
        num = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else min(start + 3000, len(text))
        block = text[start:end].strip()
        block = re.sub(r"\s+", " ", block)
        if num not in legends or len(block) > len(legends[num]):
            legends[num] = block[:2000]
    return legends


def extract_paper(pdf_path: str | Path) -> Paper:
    """Pure text extraction. No LLM."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    pages_text: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pages_text.append(page.get_text("text"))

    if not any(p.strip() for p in pages_text):
        raise RuntimeError(
            f"No text layer found in {pdf_path.name}. PaperBanana auto-pipeline does not run OCR."
        )

    pages_text = _strip_repeating_lines(pages_text)
    full_text = "\n\f\n".join(pages_text)
    flat = full_text.replace("\f", "\n")

    title = _extract_title(pages_text[0]) if pages_text else ""
    sections = _split_sections(flat)
    legends = _extract_figure_legends(flat)

    return Paper(
        title=title,
        full_text=flat,
        sections=sections,
        figure_legends=legends,
    )


REFINE_SYSTEM_PROMPT = """You prepare inputs for an academic diagram generator (PaperBanana). \
The diagram must communicate the paper's KEY FINDING or CENTRAL MESSAGE — not the research procedure.

A bad diagram says: "Phase 1: literature review → Phase 2: experiment → Phase 3: analysis."
A good diagram says: "X causes Y, unless Z is present" or "A and B interact via mechanism C, \
producing outcome D vs control E."

You will receive: (1) the Abstract, (2) the Results section, (3) the Discussion/Conclusion, \
(4) the target figure legend, and (5) the Methods section. Use them with this priority:

  PRIMARY  — Figure legend, Discussion, Abstract conclusion. These state the take-away.
  SECONDARY — Results. Source of concrete observed effects and contrasts.
  REFERENCE ONLY — Methods. Use to ground entity names, doses, reagents — never as the diagram's structure.

Return a single JSON object (no markdown fences, no commentary):
{
  "method_markdown": "Hierarchical markdown — see rules below. Despite the field name, this is the \
diagram's CONTENT, not the paper's methodology.",
  "caption": "1-3 sentence figure caption that states the take-away the figure communicates. \
Lead with the finding, not the setup.",
  "diagram_type": "Pipeline" | "Architecture" | "Workflow" | "Conceptual",
  "aspect_ratio": "16:9" | "4:3" | "1:1",
  "key_entities": ["short noun phrases that must appear as labeled nodes in the diagram"],
  "key_relations": ["A -> B", "B -| C", ...]   // -> = causes/produces, -| = inhibits/suppresses
}

=== method_markdown formatting rules ===
1. Use one `##` heading that NAMES THE FINDING, not the procedure.
     GOOD:  `## TREG co-transplantation rescues TH⁺ neurons from needle-trauma death`
     BAD:   `## Co-transplantation pipeline` / `## Research framework`
2. Use `###` sub-headings to name MECHANISMS, EFFECTS, or CONTRASTS — not workflow stages.
     GOOD:  `### Needle trauma triggers acute neuroinflammation`
            `### TREG abrogates the inflammatory infiltrate`
            `### Functional rescue: motor recovery`
     BAD:   `### Phase 1: Cell preparation`
            `### Experimental setup`
3. Inside each sub-section, use bullets or short numbered lists. Each item is an OBSERVED EFFECT, \
a CONTRAST, or a MECHANISTIC LINK — not a procedural step.
4. ALWAYS surface the paper's central contrast as explicit parallel sub-bullets when one exists \
(control vs treatment, wild-type vs mutant, before vs after, with/without an intervention). \
This is the most diagram-worthy structure.
5. Use **bold** for entities/cell types/tools/observed effects. Use *italic* for quantitative values, \
doses, time points, p-values, and conditions.
6. Include the OUTCOME (behavioral phenotype, survival %, expression change), not just the setup.
7. Aim for 12-25 lines. The planner uses hierarchy and parallelism to allocate columns and groups.

=== Example of the expected SHAPE (do NOT copy the content) ===
## TREG co-transplantation rescues TH⁺ neurons from needle-trauma death
### The hidden cost of intra-striatal grafting
- **Hamilton syringe** insertion alone triggers acute inflammation: *TNF-α↑, IL-1β↑, IFN-γ↑*, peaking *day 4*.
- The inflammation selectively kills **TH⁺ mDA neurons** (*>90% loss by week 2*), while **TH⁻ cells** mostly survive.
### Autologous TREG suppress the inflammatory cascade
- **TREG cells** isolated from the host suppress **MHCII⁺** infiltration in a dose-dependent manner.
- Effect peaks at *day 2*, persists ~7 days even though TREG dissipate within a week.
### Functional rescue
- **Control** (mDAPs alone): persistent motor deficits, *~600 TH⁺ cells* at week 20.
- **TREG co-transplant**: significant motor recovery, *~9,500 TH⁺ cells* (15× more), denser graft synapses.

=== Hard rules ===
- Output ONLY the JSON object. No code fences, no commentary.
- Do not invent details. If the source is silent on a value, omit rather than guess.
- If the paper has no clear empirical contrast (e.g., a pure-methodology or design paper), \
still center the diagram on a CONCEPT or RELATIONSHIP, not a workflow. \
Name the concept ("How formative variables drive perceived balance"), not the procedure ("Research framework").
- Prefer concrete, drawable nouns ("Hamilton syringe", "MHCII⁺ infiltrate") over abstract ones \
("methodology", "framework", "approach")."""


def _build_refine_prompt(paper: Paper, target_figure: int | None) -> str:
    abstract_text = paper.abstract[:3000] if paper.abstract else ""
    results_text = paper.results[:12000] if paper.results else ""
    discussion_text = paper.discussion[:8000] if paper.discussion else ""
    methods_text = paper.methods[:4000] if paper.methods else "(methods section not found)"

    if target_figure is None:
        candidates = []
        for num, legend in sorted(paper.figure_legends.items()):
            if re.search(r"(overview|schematic|framework|strategy|workflow|model|approach)",
                        legend, re.I):
                candidates.append((num, legend))
        if not candidates and paper.figure_legends:
            num = min(paper.figure_legends)
            candidates = [(num, paper.figure_legends[num])]
        if not candidates:
            target_legend = "(no figure legend found; build a conceptual overview from the methods text)"
            target_figure_display = "auto"
        else:
            target_figure, target_legend = candidates[0]
            target_figure_display = str(target_figure)
    else:
        if target_figure not in paper.figure_legends:
            raise ValueError(
                f"Figure {target_figure} not found. Available: {paper.available_figures()}"
            )
        target_legend = paper.figure_legends[target_figure]
        target_figure_display = str(target_figure)

    return f"""=== Paper title ===
{paper.title}

=== Target figure ===
Figure {target_figure_display}

=== Target figure legend (PRIMARY — state what the figure communicates) ===
{target_legend}

=== Discussion / Conclusion (PRIMARY — take-aways) ===
{discussion_text or "(discussion section not found)"}

=== Abstract (PRIMARY — high-level finding) ===
{abstract_text or "(abstract not found)"}

=== Results (SECONDARY — concrete observed effects) ===
{results_text or "(results section not found)"}

=== Methods (REFERENCE ONLY — entity names, doses, reagents) ===
{methods_text}
""", target_figure


async def refine_for_diagram(
    paper: Paper,
    figure_num: int | None = None,
    main_model_name: str = "",
) -> RefinedInput:
    """Single Gemini call that returns structured diagram input."""
    user_prompt, resolved_figure = _build_refine_prompt(paper, figure_num)

    cfg = ExpConfig(dataset_name="PaperBananaBench",
                    main_model_name=main_model_name)
    model = cfg.main_model_name

    contents = [{"type": "text", "text": user_prompt}]
    gen_config = types.GenerateContentConfig(
        system_instruction=REFINE_SYSTEM_PROMPT,
        temperature=0.4,
        candidate_count=1,
        max_output_tokens=4096,
        response_mime_type="application/json",
    )

    results = await call_gemini_with_retry_async(
        model_name=model,
        contents=contents,
        config=gen_config,
        max_attempts=4,
        retry_delay=5,
        error_context="pdf_ingest.refine_for_diagram",
    )
    if not results or results[0] == "Error":
        raise RuntimeError("Gemini refine call failed after retries.")

    raw = results[0]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = json_repair.loads(raw)

    return RefinedInput(
        method_markdown=str(data.get("method_markdown", "")).strip(),
        caption=str(data.get("caption", "")).strip(),
        diagram_type=str(data.get("diagram_type", "Pipeline")),
        aspect_ratio=str(data.get("aspect_ratio", "16:9")),
        key_entities=list(data.get("key_entities", [])),
        key_relations=list(data.get("key_relations", [])),
        target_figure=resolved_figure,
    )


def refined_to_dict(ri: RefinedInput) -> dict:
    return asdict(ri)


if __name__ == "__main__":
    import sys
    paper = extract_paper(sys.argv[1])
    print(f"Title: {paper.title}")
    print(f"Sections found: {list(paper.sections.keys())}")
    print(f"Figures found: {paper.available_figures()}")
    print(f"Methods length: {len(paper.methods)} chars")
    if paper.figure_legends:
        first_fig = min(paper.figure_legends)
        print(f"\nFigure {first_fig} legend (first 300 chars):")
        print(paper.figure_legends[first_fig][:300])
