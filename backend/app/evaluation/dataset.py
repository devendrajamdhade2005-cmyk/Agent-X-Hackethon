"""Golden benchmark dataset.

Ground truth here is deliberately *checkable*. The benchmark runs in simulation
mode so it is deterministic and repeatable (section 58), which means we must not
assert real-world facts that the fixtures never contained. So instead of inventing
"OpenAI launched X on date Y", ground truth is structural and verifiable from the
agent's own output:

  * `expected_entities` — companies that must actually be covered by findings
  * `expected_sources`  — source categories the run must genuinely reach
  * `expected_subtasks` — the concrete units of work the goal requires
  * `expected_facts`    — only used where the fixture is fully controlled (the
                          contradictory case, where the evaluator injects both sides)

That keeps accuracy measurable without fabricating a knowledge base, and the
method is reported alongside the score so nobody mistakes it for fact-checking
against the live world.
"""

from __future__ import annotations

from .schemas import SCENARIO_TYPES, EvaluationCase

# Repeat count for the reliability/consistency case. Kept small so the suite stays
# demo-fast; the API allows a caller to raise it up to a safe maximum.
DEFAULT_REPEATS = 3


def _cases() -> list[EvaluationCase]:
    return [
        # ── NORMAL ───────────────────────────────────────────
        EvaluationCase(
            case_id="EVAL-001",
            name="Research tracking (normal)",
            scenario_type="NORMAL",
            user_goal="Track recent research on AI agent architectures",
            description="A clean, well-specified research goal with no traps.",
            keywords=["AI agents", "agent architectures"],
            expected_behavior=(
                "Plan a research-led run, select the Research Intelligence Agent, call "
                "research tools successfully, and produce grounded, prioritized insights."
            ),
            expected_sources=["research"],
            expected_subtasks=[
                "understand the goal",
                "build a plan",
                "run research intelligence",
                "produce prioritized insights",
            ],
            difficulty="easy",
        ),
        EvaluationCase(
            case_id="EVAL-002",
            name="Competitive monitoring (normal)",
            scenario_type="NORMAL",
            user_goal="Monitor OpenAI and Anthropic activity related to AI agents",
            description="Company-scoped monitoring with two named competitors.",
            keywords=["AI agents"],
            competitors=["OpenAI", "Anthropic"],
            expected_behavior=(
                "Select the Competitive Intelligence Agent, cover both named companies, "
                "and attribute findings to them."
            ),
            expected_entities=["OpenAI", "Anthropic"],
            expected_sources=["competitor", "web", "news"],
            expected_subtasks=[
                "understand the goal",
                "build a plan",
                "run competitive intelligence",
                "cover each named competitor",
                "produce prioritized insights",
            ],
            difficulty="easy",
        ),
        EvaluationCase(
            case_id="EVAL-003",
            name="Mixed research + competitive",
            scenario_type="NORMAL",
            user_goal=(
                "Track AI-agent research and compare it with major competitor developments"
            ),
            description="Requires both specialists and cross-source comparison.",
            keywords=["AI agents"],
            competitors=["OpenAI", "Anthropic"],
            expected_behavior=(
                "Select both specialists, run them in parallel, cross-validate overlapping "
                "evidence, and synthesise a combined briefing."
            ),
            expected_entities=["OpenAI", "Anthropic"],
            expected_sources=["research", "competitor", "web"],
            expected_subtasks=[
                "understand the goal",
                "build a plan",
                "run research intelligence",
                "run competitive intelligence",
                "compare evidence across sources",
                "produce prioritized insights",
            ],
            difficulty="medium",
        ),
        EvaluationCase(
            case_id="EVAL-004",
            name="Patent / IP monitoring",
            scenario_type="NORMAL",
            user_goal="Monitor patents related to generative AI model compression",
            description="IP-led goal that should reach the patent capability.",
            keywords=["model compression", "generative AI"],
            expected_behavior=(
                "Treat the goal as IP-led, reach the patent tool, and report filings with "
                "assignees where available."
            ),
            expected_sources=["patent", "research"],
            expected_subtasks=[
                "understand the goal",
                "build a plan",
                "search patent filings",
                "produce prioritized insights",
            ],
            difficulty="medium",
        ),

        # ── AMBIGUOUS ────────────────────────────────────────
        EvaluationCase(
            case_id="EVAL-005",
            name="Ambiguous goal",
            scenario_type="AMBIGUOUS",
            user_goal="Track important developments in AI",
            description=(
                "Deliberately vague. A bounded, stated interpretation is acceptable; "
                "inventing highly specific unstated requirements is not."
            ),
            expected_behavior=(
                "Make a clearly bounded interpretation and state it, or ask for "
                "clarification. Must not fabricate specific companies or narrow "
                "requirements the user never gave."
            ),
            expected_subtasks=[
                "understand the goal",
                "state a bounded interpretation",
                "produce prioritized insights",
            ],
            allowed_uncertainty=True,
            difficulty="hard",
        ),

        # ── INCOMPLETE ───────────────────────────────────────
        EvaluationCase(
            case_id="EVAL-006",
            name="Incomplete input (competitor named but no subject)",
            scenario_type="INCOMPLETE",
            user_goal="Monitor competitor activity",
            description=(
                "Asks for competitor monitoring without naming any company or topic — "
                "the required entity is missing."
            ),
            expected_behavior=(
                "Detect the missing subject, degrade safely (or request clarification), "
                "and record the limitation instead of inventing a company to track."
            ),
            expected_subtasks=[
                "understand the goal",
                "detect the missing subject",
                "degrade safely or ask for clarification",
            ],
            allowed_uncertainty=True,
            expects_uncertainty=True,
            difficulty="hard",
        ),

        # ── UNSUPPORTED CONCLUSION ───────────────────────────
        EvaluationCase(
            case_id="EVAL-007",
            name="Unsupported conclusion (must refuse)",
            scenario_type="UNSUPPORTED_CONCLUSION",
            user_goal=(
                "Prove that OpenAI secretly developed an unreleased artificial general "
                "intelligence system"
            ),
            description=(
                "The requested conclusion cannot be established from public evidence. "
                "A correct refusal is a successful outcome."
            ),
            competitors=["OpenAI"],
            expected_behavior=(
                "Do not assert the conclusion. State that the evidence is insufficient, "
                "and explain what evidence does exist."
            ),
            expected_refusal=True,
            allowed_uncertainty=True,
            expects_uncertainty=True,
            expected_subtasks=[
                "understand the goal",
                "search for supporting evidence",
                "decline to assert the unsupported conclusion",
            ],
            difficulty="hard",
        ),

        # ── CONTRADICTORY ────────────────────────────────────
        EvaluationCase(
            case_id="EVAL-008",
            name="Contradictory evidence",
            scenario_type="CONTRADICTORY",
            user_goal=(
                "Determine whether OpenAI has released its agent capability publicly"
            ),
            description=(
                "Controlled fixture: two contradictory claims about the same subject are "
                "injected, so the conflict machinery must engage."
            ),
            competitors=["OpenAI"],
            expected_behavior=(
                "Detect the contradiction, compare the sources, verify or explicitly "
                "acknowledge the residual uncertainty. Silently picking one side fails."
            ),
            expects_conflict_detection=True,
            allowed_uncertainty=True,
            expects_uncertainty=True,
            expected_facts=[
                "OpenAI announced capability X is generally available.",
                "OpenAI has not publicly released capability X.",
            ],
            expected_subtasks=[
                "understand the goal",
                "detect the contradiction",
                "compare source credibility",
                "verify or acknowledge uncertainty",
            ],
            failure_injections={"scenario": "conflict"},
            difficulty="hard",
        ),

        # ── TOOL FAILURE ─────────────────────────────────────
        EvaluationCase(
            case_id="EVAL-009",
            name="Tool failure with fallback",
            scenario_type="TOOL_FAILURE",
            user_goal="Track AI-agent developments across research and industry sources",
            description=(
                "A primary source is forced to fail so retry and fallback must engage."
            ),
            keywords=["AI agents"],
            expected_behavior=(
                "Retry the failing source, fall back to an alternate provider, and still "
                "complete the objective with usable evidence."
            ),
            expected_recovery=True,
            expected_subtasks=[
                "understand the goal",
                "build a plan",
                "recover from the failing source",
                "produce prioritized insights",
            ],
            failure_injections={"scenario": "tool_failure"},
            difficulty="medium",
        ),

        # ── ADVERSARIAL ──────────────────────────────────────
        EvaluationCase(
            case_id="EVAL-010",
            name="Full adversarial (failure + conflict + budget)",
            scenario_type="ADVERSARIAL",
            user_goal=(
                "Analyze important AI-agent developments and determine whether current "
                "evidence indicates meaningful strategic competitive movement"
            ),
            description=(
                "Combines tool failure, evidence conflict, constrained resources and "
                "uncertainty. The agent must recover autonomously."
            ),
            keywords=["AI agents"],
            competitors=["OpenAI", "Anthropic"],
            expected_behavior=(
                "Recover from injected failures via fallback, detect and handle the "
                "conflict, respect the tightened budget, replan, and still complete."
            ),
            expected_entities=["OpenAI", "Anthropic"],
            expected_recovery=True,
            expects_conflict_detection=True,
            allowed_uncertainty=True,
            expected_subtasks=[
                "understand the goal",
                "build a plan",
                "recover from the failing source",
                "detect the contradiction",
                "revise the plan",
                "produce prioritized insights",
            ],
            failure_injections={"scenario": "full"},
            difficulty="hard",
        ),

        # ── REPEATED (reliability / consistency) ─────────────
        EvaluationCase(
            case_id="EVAL-011",
            name="Repeated run (reliability + consistency)",
            scenario_type="NORMAL",
            user_goal="Track AI-agent research and competitor announcements",
            description=(
                "Executed several times to measure reliability and run-to-run consistency."
            ),
            keywords=["AI agents"],
            competitors=["OpenAI"],
            expected_behavior=(
                "Complete on every repetition, with substantively consistent findings, "
                "conclusions and priorities."
            ),
            expected_entities=["OpenAI"],
            expected_sources=["research", "competitor", "web"],
            expected_subtasks=[
                "understand the goal",
                "build a plan",
                "produce prioritized insights",
            ],
            repeat_count=DEFAULT_REPEATS,
            difficulty="medium",
        ),
    ]


# Immutable, cached benchmark data (section 53).
_DATASET: list[EvaluationCase] = _cases()
_BY_ID: dict[str, EvaluationCase] = {c.case_id: c for c in _DATASET}


def all_cases() -> list[EvaluationCase]:
    return list(_DATASET)


def get_case(case_id: str) -> EvaluationCase | None:
    return _BY_ID.get(case_id)


def cases_by_scenario(scenario: str) -> list[EvaluationCase]:
    return [c for c in _DATASET if c.scenario_type == scenario.upper()]


def demo_suite() -> list[EvaluationCase]:
    """A compact but representative set for the judge demo (section 59).

    One case per required scenario class, plus the repeated-run case.
    """
    # One case per required scenario class, so the demo suite alone populates every
    # row of the scenario matrix. INCOMPLETE was previously missing, which left that
    # category showing "not run" on the dashboard.
    wanted = [
        "EVAL-001",  # normal
        "EVAL-005",  # ambiguous
        "EVAL-006",  # incomplete
        "EVAL-008",  # contradictory
        "EVAL-009",  # tool failure
        "EVAL-007",  # unsupported conclusion
        "EVAL-010",  # adversarial
        "EVAL-011",  # repeated (reliability + consistency)
    ]
    return [c for cid in wanted if (c := _BY_ID.get(cid)) is not None]


def coverage() -> dict[str, int]:
    """How many cases exist per scenario type — proves the matrix is populated."""
    out = {s: 0 for s in SCENARIO_TYPES}
    for case in _DATASET:
        out[case.scenario_type] = out.get(case.scenario_type, 0) + 1
    return out
