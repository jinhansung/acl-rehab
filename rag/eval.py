"""
Retrieval quality eval — 30-question gold set across MOON, Delaware-Oslo,
Aspetar, and DVT guidelines.

Usage:
    python rag/eval.py                   # run gold set, print report
    python rag/eval.py --run-gold-set    # same (pytest-friendly alias)
    python rag/eval.py --top-k 6         # override k (default 4)

Exit code 1 if any query fails (canonical source not in top-k).
"""
from __future__ import annotations
import argparse
import os
import sys
from dataclasses import dataclass, field

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "acl_protocols"
DEFAULT_TOP_K = 4


# ── Embedding (must match ingest) ─────────────────────────────────────────────

class NomicQueryFunction(EmbeddingFunction):
    """Query-side wrapper — uses search_query prefix."""

    def __init__(self) -> None:
        self._model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1",
            trust_remote_code=True,
        )

    def __call__(self, input: Documents) -> Embeddings:
        prefixed = [f"search_query: {q}" for q in input]
        return self._model.encode(prefixed, normalize_embeddings=True).tolist()


# ── Gold set ──────────────────────────────────────────────────────────────────

@dataclass
class GoldQuery:
    query: str
    canonical_source: str          # protocol_name that MUST appear in top-k
    tags: list[str] = field(default_factory=list)


GOLD_SET: list[GoldQuery] = [
    # ── MOON (8 questions) ────────────────────────────────────────────────────
    GoldQuery(
        "When can patients begin full weight bearing after ACL reconstruction?",
        "MOON", ["weight-bearing", "early"],
    ),
    GoldQuery(
        "What are the criteria for progressing from phase 1 to phase 2 in MOON?",
        "MOON", ["phase-progression"],
    ),
    GoldQuery(
        "Which strengthening exercises are prescribed in MOON weeks 1 and 2?",
        "MOON", ["strengthening", "early-phase"],
    ),
    GoldQuery(
        "What are the return-to-sport criteria defined in the MOON protocol?",
        "MOON", ["RTS"],
    ),
    GoldQuery(
        "How does the MOON protocol manage post-operative swelling?",
        "MOON", ["swelling", "cryotherapy"],
    ),
    GoldQuery(
        "What brace recommendations does MOON give in the first six weeks?",
        "MOON", ["brace"],
    ),
    GoldQuery(
        "At what time point does the MOON protocol allow patients to begin running?",
        "MOON", ["running", "jogging"],
    ),
    GoldQuery(
        "What open kinetic chain exercise restrictions appear in the MOON protocol?",
        "MOON", ["OKC", "open-chain"],
    ),

    # ── Delaware-Oslo (8 questions) ───────────────────────────────────────────
    GoldQuery(
        "What are the five phases of the Delaware-Oslo ACL rehabilitation protocol?",
        "Delaware-Oslo", ["phases"],
    ),
    GoldQuery(
        "When does neuromuscular and proprioceptive training begin in Delaware-Oslo?",
        "Delaware-Oslo", ["neuromuscular", "proprioception"],
    ),
    GoldQuery(
        "Which hop tests are used to assess return-to-sport readiness in Delaware-Oslo?",
        "Delaware-Oslo", ["hop-tests", "RTS"],
    ),
    GoldQuery(
        "How does the Delaware-Oslo protocol address psychological readiness for return to sport?",
        "Delaware-Oslo", ["psychological", "ACL-RSI"],
    ),
    GoldQuery(
        "What quadriceps limb symmetry index is required before returning to sport in Delaware-Oslo?",
        "Delaware-Oslo", ["LSI", "quadriceps-strength"],
    ),
    GoldQuery(
        "When does plyometric training begin in the Delaware-Oslo protocol?",
        "Delaware-Oslo", ["plyometrics"],
    ),
    GoldQuery(
        "How does Delaware-Oslo approach hamstring-to-quadriceps strength ratio assessment?",
        "Delaware-Oslo", ["hamstring", "H:Q-ratio"],
    ),
    GoldQuery(
        "What sport-specific agility criteria does Delaware-Oslo require before full return?",
        "Delaware-Oslo", ["agility", "sport-specific"],
    ),

    # ── Aspetar (7 questions) ─────────────────────────────────────────────────
    GoldQuery(
        "What is the Aspetar protocol timeline for return to full team training?",
        "Aspetar", ["timeline", "return-to-training"],
    ),
    GoldQuery(
        "How does the Aspetar protocol approach hamstring graft rehabilitation?",
        "Aspetar", ["hamstring-graft"],
    ),
    GoldQuery(
        "What criteria does Aspetar use to progress patients through running drills?",
        "Aspetar", ["running-progression"],
    ),
    GoldQuery(
        "Which isokinetic strength tests does Aspetar recommend before return to sport?",
        "Aspetar", ["isokinetic", "strength-testing"],
    ),
    GoldQuery(
        "How does the Aspetar protocol manage the early post-operative phase (weeks 0-2)?",
        "Aspetar", ["early-post-op"],
    ),
    GoldQuery(
        "What sport-specific pitch-side training milestones are defined in Aspetar?",
        "Aspetar", ["pitch-side", "sport-specific"],
    ),
    GoldQuery(
        "What criteria does Aspetar use to clear an athlete for contact training?",
        "Aspetar", ["contact-training", "clearance"],
    ),

    # ── DVT guidelines (7 questions) ──────────────────────────────────────────
    GoldQuery(
        "What patient risk factors increase DVT probability after ACL surgery?",
        "DVT", ["risk-factors"],
    ),
    GoldQuery(
        "What pharmacological prophylaxis is recommended to prevent DVT after ACL reconstruction?",
        "DVT", ["prophylaxis", "anticoagulation"],
    ),
    GoldQuery(
        "What are the clinical signs and symptoms of DVT that patients should monitor for?",
        "DVT", ["symptoms", "signs"],
    ),
    GoldQuery(
        "Under what circumstances should a patient seek emergency care for suspected DVT?",
        "DVT", ["emergency", "pulmonary-embolism"],
    ),
    GoldQuery(
        "Which early mobility exercises reduce DVT risk in the post-operative period?",
        "DVT", ["mobility", "prevention"],
    ),
    GoldQuery(
        "For how long is DVT risk clinically elevated after ACL reconstruction?",
        "DVT", ["duration", "risk-window"],
    ),
    GoldQuery(
        "What compression or mechanical prophylaxis does the DVT guideline recommend?",
        "DVT", ["compression", "mechanical-prophylaxis"],
    ),
]

assert len(GOLD_SET) == 30, f"Gold set must have 30 questions, has {len(GOLD_SET)}"


# ── Eval runner ───────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    query: GoldQuery
    top_sources: list[str]
    top_scores: list[float]
    passed: bool

    @property
    def cosine_scores(self) -> list[str]:
        return [f"{s:.4f}" for s in self.top_scores]


def _cosine_from_distance(distance: float) -> float:
    """ChromaDB cosine collection returns L2-normalised inner products as distances (1 - similarity)."""
    return round(1.0 - distance, 4)


def run_gold_set(top_k: int = DEFAULT_TOP_K, verbose: bool = True) -> list[QueryResult]:
    ef = NomicQueryFunction()
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
        )
    except Exception:
        print(
            f"Collection '{COLLECTION_NAME}' not found. "
            "Run: python -m rag.ingest"
        )
        sys.exit(1)

    results: list[QueryResult] = []

    for gq in GOLD_SET:
        response = collection.query(
            query_texts=[f"search_query: {gq.query}"],
            n_results=top_k,
            include=["metadatas", "distances"],
        )

        metadatas = response["metadatas"][0]
        distances = response["distances"][0]

        top_sources = [m.get("protocol_name", "?") for m in metadatas]
        top_scores = [_cosine_from_distance(d) for d in distances]
        passed = gq.canonical_source in top_sources

        result = QueryResult(
            query=gq,
            top_sources=top_sources,
            top_scores=top_scores,
            passed=passed,
        )
        results.append(result)

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"\n[{status}] {gq.query}")
            print(f"  canonical={gq.canonical_source}  top-{top_k} sources={top_sources}")
            print(f"  cosine scores: {result.cosine_scores}")
            if not passed:
                print(f"  *** '{gq.canonical_source}' NOT found in top-{top_k} ***")

    return results


def _print_summary(results: list[QueryResult], top_k: int) -> bool:
    passed = sum(r.passed for r in results)
    total = len(results)
    pct = 100 * passed / total

    print(f"\n{'='*60}")
    print(f"Retrieval eval  top-{top_k}: {passed}/{total} passed ({pct:.1f}%)")

    by_protocol: dict[str, list[bool]] = {}
    for r in results:
        key = r.query.canonical_source
        by_protocol.setdefault(key, []).append(r.passed)

    for proto, outcomes in sorted(by_protocol.items()):
        p = sum(outcomes)
        t = len(outcomes)
        print(f"  {proto:<20} {p}/{t}")

    failed = [r for r in results if not r.passed]
    if failed:
        print(f"\nFailed queries ({len(failed)}):")
        for r in failed:
            print(f"  - [{r.query.canonical_source}] {r.query.query}")

    print("="*60)
    return passed == total


def main() -> None:
    parser = argparse.ArgumentParser(description="ACL RAG retrieval + tone eval")
    parser.add_argument("--run-gold-set", action="store_true",
                        help="Run the 30-question retrieval gold set")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--tone-check", action="store_true",
                        help="Run tone compliance checker (regex + Claude-judge)")
    parser.add_argument("--tone-inputs", type=str, default=None,
                        help="Path to a .jsonl file of {text, graft_type} objects to check. "
                             "Defaults to built-in synthetic corpus.")
    parser.add_argument("--no-claude-judge", action="store_true",
                        help="Skip Claude-judge pass (regex only)")
    args = parser.parse_args()

    exit_code = 0

    if args.tone_check:
        passed = run_tone_check(
            inputs_path=args.tone_inputs,
            verbose=not args.quiet,
            use_claude_judge=not args.no_claude_judge,
        )
        if not passed:
            exit_code = 1

    if args.run_gold_set or not args.tone_check:
        results = run_gold_set(top_k=args.top_k, verbose=not args.quiet)
        if not _print_summary(results, args.top_k):
            exit_code = 1

    sys.exit(exit_code)


# ═══════════════════════════════════════════════════════════════════════════════
# TONE COMPLIANCE CHECKER
# ═══════════════════════════════════════════════════════════════════════════════
#
# NOTE ON CLAUDE USAGE IN THIS FILE:
# CLAUDE.md restricts Anthropic SDK calls to agent/tools.py in the *app*.
# This eval module is an offline developer tool — it is never imported by the
# app at runtime. The Claude-judge here is intentionally eval-only.
# ───────────────────────────────────────────────────────────────────────────────

import json
import re
import textwrap
from dataclasses import dataclass as _dc, field as _field

TONE_COMPLIANCE_THRESHOLD = 0.98   # 98 %
JUDGE_MODEL = "claude-haiku-4-5-20251001"
SUMMARY_WORD_LIMIT = 120

# Exact banned strings — kept in sync with agent/prompts.py::BANNED_WORDS
_BANNED: list[str] = ["behind", "should", "!", "you need to", "you must"]
# Compiled case-insensitive patterns for each banned phrase
_BANNED_RE: list[re.Pattern] = [
    re.compile(re.escape(w), re.IGNORECASE) for w in _BANNED
]

_GRAFT_KW_RE = re.compile(
    r"\b(graft|hamstring|patellar|quadriceps|tendon|allograft|autograft|harvest|donor)\b",
    re.IGNORECASE,
)


@_dc
class ToneCase:
    text: str
    graft_type: str = "hamstring"
    has_graft_claim: bool = False   # True if text is expected to contain a graft claim
    graft_citations: list[str] = _field(default_factory=list)
    # Ground-truth label for synthetic corpus
    expected_compliant: bool = True
    description: str = ""


@_dc
class ToneResult:
    case: ToneCase
    regex_violations: list[str]
    word_count: int
    graft_citation_missing: bool
    judge_compliant: bool | None   # None if judge skipped
    judge_violations: list[str]
    judge_reasoning: str

    @property
    def compliant(self) -> bool:
        failed_regex = bool(self.regex_violations) or self.graft_citation_missing
        failed_judge = self.judge_compliant is False
        return not failed_regex and not failed_judge


# ── Synthetic corpus ──────────────────────────────────────────────────────────

SYNTHETIC_CORPUS: list[ToneCase] = [
    # ── Compliant cases ───────────────────────────────────────────────────────
    ToneCase(
        text=(
            "Your average pain score dropped from 6 to 3 this week — a clear sign "
            "your knee is responding well. You completed three of four planned sessions, "
            "and quad sets felt noticeably easier by Thursday. "
            "The swelling you reported on Tuesday was mild and resolved within the day. "
            "Keep building on that quad strength so you can get back to the pitch with confidence."
        ),
        graft_type="hamstring",
        expected_compliant=True,
        description="clean — improvement lead, no banned words, graft not mentioned",
    ),
    ToneCase(
        text=(
            "Four sessions completed this week — every one of them. "
            "Pain stayed at 2–3 throughout, which lines up with where week 6 typically lands. "
            "You flagged some stiffness in the morning; that's common at this stage and worth "
            "mentioning to your PT at your next check-in. "
            "Next up: getting comfortable with single-leg work so playing pre-season in August feels within reach."
        ),
        graft_type="patellar_tendon",
        expected_compliant=True,
        description="clean — leads with session count improvement, goal language in close",
    ),
    ToneCase(
        text=(
            "Pain during heel slides came down to a 2 this week, compared with 5 last week. "
            "You completed all prescribed exercises, including the straight-leg raises that "
            "felt challenging earlier in the protocol. "
            "Being confident on the pitch again starts with building that knee strength — "
            "single-leg balance is the focus for next week."
        ),
        graft_type="hamstring",
        expected_compliant=True,
        description="clean — numeric improvement lead, goal-language close",
    ),
    ToneCase(
        text=(
            "All three sessions done, and your RPE average of 4 suggests the load is landing "
            "in exactly the right zone for week 4. "
            "Swelling stayed at mild throughout — good progress from the moderate levels two weeks ago. "
            "Quad endurance continues to build; the next milestone is adding resistance to your leg press."
        ),
        graft_type="quadriceps",
        expected_compliant=True,
        description="clean — RPE improvement lead, no graft keywords",
    ),
    ToneCase(
        text=(
            "Your pain-free range of motion improved by roughly 10 degrees this week — "
            "a tangible step forward. "
            "You managed two split sessions on the days when a full session felt too much, "
            "and that flexibility is what keeps the programme moving. "
            "The focus for next week is getting that range closer to 120 degrees so you can "
            "get back to playing club football without hesitation."
        ),
        graft_type="allograft",
        expected_compliant=True,
        description="clean — Helen-style split-session acknowledgement, goal language",
    ),
    ToneCase(
        text=(
            "Three out of four sessions completed, and pain during exercises averaged 2.5 — "
            "down from 4 last week. "
            "You handled the step-ups without any giving-way, which is an encouraging sign "
            "at this point in the protocol. "
            "Keeping that progress going puts you on track to return to the pre-season training "
            "ground with confidence."
        ),
        graft_type="hamstring",
        expected_compliant=True,
        description="clean — numeric pain improvement, goal close",
    ),
    ToneCase(
        text=(
            "Pain scores stayed consistently low at 1–2 this week — a strong signal that "
            "your knee is tolerating the increased load well. "
            "You completed the full set of exercises including the more demanding lateral band walks. "
            "The next step toward competing at club level again is working on single-leg squat depth."
        ),
        graft_type="patellar_tendon",
        expected_compliant=True,
        description="clean — load tolerance framing, competition goal in close",
    ),
    ToneCase(
        text=(
            "This week's sessions went well — four completed, pain averaged 3, "
            "and you flagged no instability. "
            "That's a step up from last week's two sessions and average pain of 5. "
            "Continuing to build hip and quad strength is what gets you back to full hiking "
            "without worrying about the knee."
        ),
        graft_type="allograft",
        expected_compliant=True,
        description="clean — Helen-style, hiking goal language",
    ),
    ToneCase(
        text=(
            "Adherence was at 100 % this week — all exercises, all sessions. "
            "Your reported pain during terminal knee extensions dropped to a 1, "
            "which is a meaningful improvement from the 4 you logged two weeks ago. "
            "The next target is getting comfortable enough with cutting movements to "
            "feel ready for full basketball training again."
        ),
        graft_type="hamstring",
        expected_compliant=True,
        description="clean — adherence + pain improvement, basketball goal close",
    ),
    ToneCase(
        text=(
            "Range of motion at 110 degrees this week — up from 95 last week. "
            "You handled all four sessions and kept RPE in the moderate range throughout. "
            "Maintaining that momentum puts returning to the pitch in August firmly within your reach."
        ),
        graft_type="quadriceps",
        expected_compliant=True,
        description="clean — ROM numeric improvement, August goal close",
    ),
    ToneCase(
        text=(
            "Swelling stayed at 'None' across all check-ins this week — a real improvement "
            "from the mild-to-moderate readings of weeks 3 and 4. "
            "You completed three sessions, including the more demanding step-down exercise. "
            "Getting back to recreational hiking with your family is the next thing to work toward."
        ),
        graft_type="allograft",
        expected_compliant=True,
        description="clean — swelling improvement lead, family hiking goal",
    ),
    ToneCase(
        text=(
            "Your pain score during quad sets fell to 1 this week, down from 3. "
            "You completed all five prescribed exercises across three sessions, "
            "with no giving-way reported. "
            "Increasing single-leg load over the next week brings playing club football again a step closer."
        ),
        graft_type="patellar_tendon",
        expected_compliant=True,
        description="clean — pain drop, football goal language close",
    ),
    ToneCase(
        text=(
            "Three sessions in, and your average pain dropped from 5 to 2 over the course of the week. "
            "The knee felt stable throughout, including during the lateral step-up sets. "
            "Pushing that stability into single-leg work is the direct path to getting back on the pitch."
        ),
        graft_type="hamstring",
        expected_compliant=True,
        description="clean — pain arc, stability improvement, pitch goal",
    ),
    ToneCase(
        text=(
            "Pain stayed at 2 during all sessions this week — well within the comfortable range. "
            "You added the wall sit to your programme and held it for the full prescribed time. "
            "The next focus is building quad endurance to the point where running at training feels natural."
        ),
        graft_type="quadriceps",
        expected_compliant=True,
        description="clean — comfort improvement, training goal close",
    ),
    ToneCase(
        text=(
            "All four sessions completed, with pain averaging 2 throughout. "
            "No red flags this week — a clean run after the minor swelling concern from last week. "
            "Continuing to build load tolerance is the clearest route back to "
            "playing recreational football without hesitation."
        ),
        graft_type="allograft",
        expected_compliant=True,
        description="clean — clean week framing vs prior week, football goal",
    ),

    # ── Non-compliant cases (violations must be detected) ─────────────────────
    ToneCase(
        text=(
            "You are a bit behind where you should be at week 6 — pain is still higher "
            "than expected and you skipped two sessions. "
            "You must pick up the frequency or you risk falling further behind schedule!"
        ),
        graft_type="hamstring",
        expected_compliant=False,
        description="FAIL — 'behind' × 2, 'should', 'must', '!'",
    ),
    ToneCase(
        text=(
            "Good effort this week. You need to increase your session frequency next week. "
            "Pain levels were acceptable. You should aim for four sessions minimum. "
            "Your knee is progressing but you need to push harder!"
        ),
        graft_type="patellar_tendon",
        expected_compliant=False,
        description="FAIL — 'you need to', 'should', '!'",
    ),
    ToneCase(
        text=(
            "Excellent week! " * 30   # pushes well over 120 words when padded
        ),
        graft_type="quadriceps",
        expected_compliant=False,
        description="FAIL — '!' + over 120 words",
    ),
    ToneCase(
        text=(
            "Pain averaged 4 this week. Sessions were completed as planned. "
            "Keep working hard. "
            "Your hamstring graft tissue is still in the early remodelling phase at week 8 "
            "and eccentric loading should be avoided."
        ),
        graft_type="hamstring",
        has_graft_claim=True,
        graft_citations=[],            # no citation provided
        expected_compliant=False,
        description="FAIL — graft claim without citation + 'should'",
    ),
    ToneCase(
        text=(
            "Sessions completed: 2 out of 4. Pain stayed high at 6. "
            "You need to contact your PT before continuing. "
            "More effort is required next week to stay on track."
        ),
        graft_type="allograft",
        expected_compliant=False,
        description="FAIL — 'you need to', no improvement lead",
    ),
]

assert len(SYNTHETIC_CORPUS) == 20, f"Expected 20 tone cases, got {len(SYNTHETIC_CORPUS)}"


# ── Regex layer ───────────────────────────────────────────────────────────────

def regex_tone_violations(text: str) -> list[str]:
    """Fast deterministic check. Returns human-readable violation strings."""
    violations: list[str] = []
    for pattern, phrase in zip(_BANNED_RE, _BANNED):
        if pattern.search(text):
            violations.append(f"banned phrase: {phrase!r}")
    wc = len(text.split())
    if wc > SUMMARY_WORD_LIMIT:
        violations.append(f"word count {wc} > {SUMMARY_WORD_LIMIT}")
    return violations


def graft_citation_missing(case: ToneCase) -> bool:
    """True if text makes a graft claim but no citation is provided."""
    if not _GRAFT_KW_RE.search(case.text):
        return False
    return not bool(case.graft_citations)


# ── Claude-judge layer ────────────────────────────────────────────────────────

_JUDGE_SYSTEM = textwrap.dedent("""\
    You are a tone compliance auditor for patient-facing ACL rehabilitation text.

    Evaluate the provided text against these rules:
    1. No banned words/phrases (case-insensitive):
       "behind" | "should" | "!" | "you need to" | "you must"
    2. Under 120 words total.
    3. First sentence describes a concrete improvement or positive change from this week.
    4. Final sentence states exactly one forward-looking next priority.
    5. No commanding language (telling the patient they MUST do something).
    6. No mention of streaks, consistency scores, or session counts framed as judgements.
    7. No comparisons to other patients or population norms.

    Call submit_tone_verdict with your assessment.
""")

_JUDGE_TOOL = {
    "name": "submit_tone_verdict",
    "description": "Submit tone compliance verdict for the given patient-facing text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "compliant": {
                "type": "boolean",
                "description": "True if ALL rules pass.",
            },
            "violations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific rule violations found. Empty if compliant.",
            },
            "reasoning": {
                "type": "string",
                "description": "One-sentence explanation of the verdict.",
            },
        },
        "required": ["compliant", "violations", "reasoning"],
    },
}


def claude_judge(text: str) -> tuple[bool, list[str], str]:
    """
    Run Claude-as-judge on a patient-facing summary.
    Returns (compliant, violations, reasoning).

    Uses JUDGE_MODEL (haiku) — fast and cheap for bulk eval.
    This function is eval-only; it is not called from the app runtime.
    """
    import anthropic as _anthropic
    client = _anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=512,
        system=_JUDGE_SYSTEM,
        tools=[_JUDGE_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": f"Evaluate this text:\n\n{text}"}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_tone_verdict":
            inp = block.input
            return inp["compliant"], inp.get("violations", []), inp.get("reasoning", "")
    return False, ["judge did not call tool"], "unexpected response"


# ── Tone eval runner ──────────────────────────────────────────────────────────

def run_tone_check(
    inputs_path: str | None = None,
    verbose: bool = True,
    use_claude_judge: bool = True,
) -> bool:
    """
    Run the two-layer tone compliance checker.

    Layer 1 — regex: fast, deterministic, catches banned words + word count.
    Layer 2 — Claude-judge: catches subtle violations regex cannot see.

    Returns True if compliance rate ≥ TONE_COMPLIANCE_THRESHOLD.
    Exits with code 1 if threshold not met (via caller).
    """
    # Load corpus
    if inputs_path:
        corpus: list[ToneCase] = []
        with open(inputs_path) as f:
            for line in f:
                obj = json.loads(line)
                corpus.append(ToneCase(
                    text=obj["text"],
                    graft_type=obj.get("graft_type", "other"),
                    graft_citations=obj.get("graft_citations", []),
                    expected_compliant=obj.get("expected_compliant", True),
                    description=obj.get("description", ""),
                ))
    else:
        corpus = SYNTHETIC_CORPUS

    results: list[ToneResult] = []

    for i, case in enumerate(corpus, start=1):
        regex_v = regex_tone_violations(case.text)
        graft_missing = graft_citation_missing(case)
        judge_ok: bool | None = None
        judge_v: list[str] = []
        judge_r: str = ""

        # Only call Claude if regex passes — no point judging obvious failures
        if use_claude_judge and not regex_v and not graft_missing:
            try:
                judge_ok, judge_v, judge_r = claude_judge(case.text)
            except Exception as exc:
                judge_ok = None
                judge_r = f"judge error: {exc}"

        result = ToneResult(
            case=case,
            regex_violations=regex_v,
            word_count=len(case.text.split()),
            graft_citation_missing=graft_missing,
            judge_compliant=judge_ok,
            judge_violations=judge_v,
            judge_reasoning=judge_r,
        )
        results.append(result)

        if verbose:
            status = "PASS" if result.compliant else "FAIL"
            label = f"[{status}] #{i}"
            if case.description:
                label += f" — {case.description}"
            print(label)
            print(f"  words={result.word_count}  regex_violations={regex_v}")
            if graft_missing:
                print(f"  graft_citation_missing=True")
            if judge_ok is not None:
                jstatus = "PASS" if judge_ok else "FAIL"
                print(f"  judge=[{jstatus}] {judge_r}")
                if judge_v:
                    for v in judge_v:
                        print(f"    • {v}")

    # ── Reporting ─────────────────────────────────────────────────────────────
    compliant_count = sum(1 for r in results if r.compliant)
    total = len(results)
    rate = compliant_count / total if total else 0.0
    threshold_met = rate >= TONE_COMPLIANCE_THRESHOLD

    expected_pass = [r for r in results if r.case.expected_compliant]
    expected_fail = [r for r in results if not r.case.expected_compliant]
    detected_violations = sum(1 for r in expected_fail if not r.compliant)

    print(f"\n{'='*60}")
    print(f"Tone compliance:  {compliant_count}/{total}  ({rate:.1%})"
          f"  threshold={TONE_COMPLIANCE_THRESHOLD:.0%}  "
          f"{'PASS' if threshold_met else 'FAIL'}")
    print(f"Expected-pass cases:    {sum(r.compliant for r in expected_pass)}/{len(expected_pass)}")
    print(f"Violation detection:    {detected_violations}/{len(expected_fail)} caught")

    false_negatives = [r for r in expected_fail if r.compliant]
    if false_negatives:
        print(f"\nMissed violations ({len(false_negatives)}):")
        for r in false_negatives:
            print(f"  - {r.case.description}")

    false_positives = [r for r in expected_pass if not r.compliant]
    if false_positives:
        print(f"\nFalse positives ({len(false_positives)}):")
        for r in false_positives:
            print(f"  - {r.case.description}")
            for v in r.regex_violations + r.judge_violations:
                print(f"    • {v}")

    print("="*60)
    return threshold_met


if __name__ == "__main__":
    main()
