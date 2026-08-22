"""Task 6 — evaluation layer tests.

The 17 checks from the brief (section 51). Metric maths and classification are tested
directly with hand-built fixtures (fast, deterministic); case execution and the suite
paths drive the real agent in simulation mode.

`store.reset()` is called where a test depends on a clean history, because the store
is a module global shared across the process — the same convention the rest of the
suite uses.
"""

from __future__ import annotations

import asyncio

from app.evaluation import dataset, human, metrics as M, store
from app.evaluation.automated import (
    ClaimExtractor,
    ConsistencyEvaluator,
    EvidenceEvaluator,
    GroundednessEvaluator,
    HallucinationEvaluator,
    RecoveryEvaluator,
    RefusalEvaluator,
    ReliabilityEvaluator,
    RobustnessEvaluator,
    TaskCompletionEvaluator,
    UncertaintyEvaluator,
)
from app.evaluation.engine import EvaluationEngine
from app.evaluation.regression import compare_with_previous
from app.evaluation.runner import SuiteRunner
from app.evaluation.schemas import (
    CLAIM_SUPPORTED,
    CLAIM_UNSUPPORTED,
    KIND_FACTUAL,
    KIND_HYPOTHESIS,
    KIND_RECOMMENDATION,
    EvaluationCase,
    HumanEvaluation,
    Thresholds,
)


# ── fixtures ─────────────────────────────────────────────────
def finding(fid="f1", title="OpenAI launches an AI agents platform", **kw):
    base = {
        "id": fid, "title": title, "source": "competitor",
        "summary": title, "url": f"https://techcrunch.com/{fid}",
        "published_date": "2026-08-01", "provider": "newsapi", "tool": "competitor_search",
        "competitor": "OpenAI", "credibility": "standard", "simulated": False,
        "signals": ["launch"], "relevance": 0.7, "corroborated_by": [],
        "meta": {},
    }
    base.update(kw)
    return base


def insight(fid="f1", **kw):
    base = {
        "id": f"ins_{fid}", "finding_id": fid,
        "title": "OpenAI launches an AI agents platform",
        "what_happened": "OpenAI launches an AI agents platform",
        "summary": "OpenAI launches an AI agents platform for developers",
        "why_it_matters": "This matters because OpenAI is on the tracked list",
        "priority": "HIGH",
        "recommended_action": "Brief the product team this week",
        "source": "News", "source_url": "https://techcrunch.com/f1",
        "provider": "newsapi", "published_date": "2026-08-01",
        "competitor": "OpenAI", "signals": ["launch"], "confidence": "standard",
        "score": 0.7, "simulated": False, "author": "heuristic-analyst",
    }
    base.update(kw)
    return base


def run(findings=None, insights=None, framework=None, metrics=None, **kw):
    base = {
        "status": "completed",
        "run_id": "r1",
        "goal": "Track AI agents and OpenAI",
        "findings": findings if findings is not None else [finding()],
        "insights": insights if insights is not None else [insight()],
        "summary": "A briefing about OpenAI and AI agents.",
        "execution_plan": [{"agent": "competitive_agent", "selected": True}],
        "framework": {
            "plan_version": 1, "replan_count": 0, "verify_count": 0,
            "completed_agents": ["competitive_agent"], "tool_executions": [
                {"tool_name": "competitor_search", "status": "ok", "latency_ms": 120, "attempt": 1}
            ],
            "tool_errors": [], "fallback_history": [], "conflicting_evidence": [],
            "uncertainty_flags": [], "verification_status": "not_started",
            "hypotheses": [], "overall_confidence": 0.75, "injected_events": [],
            "resource": {"tool_calls": 1, "max_tool_calls": 14, "llm_calls": 0,
                         "estimated_cost": 0.002, "elapsed_ms": 400},
            **(framework or {}),
        },
        "metrics": {
            "duration_ms": 400, "tool_calls": 1, "llm_calls": 0,
            "findings_relevant": 1, "insights": 1, "parallel_agents": 1,
            "estimated_cost": 0.002, **(metrics or {}),
        },
    }
    base.update(kw)
    return base


def simple_case(**kw) -> EvaluationCase:
    base = dict(
        case_id="T-1", name="test", scenario_type="NORMAL",
        user_goal="Track AI agents and OpenAI",
        expected_subtasks=["understand the goal", "produce prioritized insights"],
    )
    base.update(kw)
    return EvaluationCase(**base)  # type: ignore[arg-type]


# ── 1. Metric calculations ───────────────────────────────────
def test_01_metric_calculations():
    d = M.distribution([100.0, 200.0, 300.0])
    assert d["mean"] == 200.0 and d["median"] == 200.0
    assert d["p95"] is None and "needs >=5" in d["p95_note"]      # honest about sample size
    d5 = M.distribution([1.0, 2.0, 3.0, 4.0, 5.0])
    assert d5["p95"] is not None
    assert M.mean_of([0.5, None, 1.0]) == 0.75
    assert M.mean_of([None, None]) is None
    # Catalogue is complete and self-describing.
    assert len(M.catalogue_dicts()) == 14
    for spec in M.CATALOGUE.values():
        assert spec.definition and spec.formula and spec.unit and spec.data_source
    assert M.spec(M.HALLUCINATION_RATE).higher_is_better is False


# ── 2. Case execution ───────────────────────────────────────
def test_02_case_execution_uses_the_real_agent():
    case = dataset.get_case("EVAL-001")
    ev, raw = asyncio.run(EvaluationEngine().evaluate_case(case, simulation_mode=True))
    assert ev.status == "completed"
    assert ev.agent_run_id, "must record the real agent run id it measured"
    assert raw.get("findings"), "the agent actually collected evidence"
    assert ev.provenance["framework_version"]
    assert ev.provenance["simulation_mode"] is True


# ── 3. Groundedness classification ──────────────────────────
def test_03_groundedness_classification():
    ex = ClaimExtractor()
    # Supported: the claim's content is in the cited evidence.
    claims = ex.extract(run())
    factual = [c for c in claims if c.kind == KIND_FACTUAL]
    assert factual and all(c.verdict == CLAIM_SUPPORTED for c in factual)
    g = GroundednessEvaluator().evaluate(simple_case(), claims)
    assert g.available and g.value == 1.0

    # Unsupported: the insight cites a finding that does not exist.
    orphan = run(insights=[insight(fid="missing", what_happened="A totally different event occurred")])
    bad = ex.extract(orphan)
    assert any(c.verdict == CLAIM_UNSUPPORTED for c in bad if c.kind == KIND_FACTUAL)
    g2 = GroundednessEvaluator().evaluate(simple_case(), bad)
    assert g2.value < 1.0


# ── 4. Hallucination classification ─────────────────────────
def test_04_hallucination_classification():
    ex = ClaimExtractor()
    orphan = run(insights=[insight(
        fid="", what_happened="Acme Corp acquired Globex for $9 billion",
        summary="Acme Corp acquired Globex for $9 billion",
    )])
    h = HallucinationEvaluator().evaluate(simple_case(), ex.extract(orphan))
    assert h.available and h.value > 0.0
    d = h.details
    # Recommendations and labelled hypotheses must never be counted as hallucinations.
    assert d["recommendations_excluded"] >= 1
    hyp_run = run(framework={"hypotheses": [
        {"hypothesis_id": "h1", "statement": "Momentum is shifting", "status": "PARTIALLY_SUPPORTED"}
    ]})
    claims = ex.extract(hyp_run)
    assert any(c.kind == KIND_HYPOTHESIS for c in claims)
    assert any(c.kind == KIND_RECOMMENDATION for c in claims)
    h2 = HallucinationEvaluator().evaluate(simple_case(), claims)
    assert h2.value == 0.0
    assert h2.details["labelled_hypotheses_excluded"] >= 1


# ── 5. Task completion ──────────────────────────────────────
def test_05_task_completion():
    case = simple_case(expected_subtasks=[
        "understand the goal", "run competitive intelligence",
        "run research intelligence", "produce prioritized insights",
    ])
    t = TaskCompletionEvaluator().evaluate(case, run())
    assert t.available
    # 3 of 4: the research agent never ran.
    assert t.details["required_subtask_count"] == 4
    assert t.details["completed_subtask_count"] == 3
    assert abs(t.value - 0.75) < 1e-6
    assert t.details["status"] == "PARTIAL"
    # An HTTP-200-shaped run with no insights must not count as completion.
    empty = run(insights=[], findings=[])
    t2 = TaskCompletionEvaluator().evaluate(
        simple_case(expected_subtasks=["produce prioritized insights"]), empty
    )
    assert t2.value == 0.0


# ── 6. Recovery scoring ─────────────────────────────────────
def test_06_recovery_scoring():
    recovered = run(framework={
        "injected_events": [{"type": "tool_failed", "tool": "web_search"}],
        "fallback_history": [{"tool": "web_search", "from": "tavily", "to": "news_search",
                              "recovered": True}],
        "tool_errors": [{"tool": "web_search", "error": "injected failure"}],
    })
    r = RecoveryEvaluator().evaluate(simple_case(expected_recovery=True), recovered)
    assert r.available and r.value == 1.0
    assert r.details["fallback_success_rate"] == 1.0

    # Partial output with no recovery attempt must not score as recovery.
    not_recovered = run(framework={
        "injected_events": [{"type": "tool_failed", "tool": "web_search"}],
        "fallback_history": [], "tool_errors": [{"tool": "web_search", "error": "boom"}],
    })
    r2 = RecoveryEvaluator().evaluate(simple_case(expected_recovery=True), not_recovered)
    assert r2.value == 0.0

    # No injection at all → unavailable, not a zero.
    r3 = RecoveryEvaluator().evaluate(simple_case(), run())
    assert r3.available is False and r3.unavailable_reason


# ── 7. Repeated-run consistency ─────────────────────────────
def test_07_repeated_run_consistency():
    a = run()
    b = run()
    c = ConsistencyEvaluator().evaluate([a, b])
    assert c.available and c.value == 1.0

    different = run(
        findings=[finding(fid="zz", title="Something else entirely")],
        insights=[insight(fid="zz", priority="LOW")],
        framework={"overall_confidence": 0.2},
        status="completed_partial",
    )
    c2 = ConsistencyEvaluator().evaluate([a, different])
    assert c2.value < 0.5
    # A single run cannot be assessed for consistency.
    assert ConsistencyEvaluator().evaluate([a]).available is False


# ── 8. Baseline comparison ──────────────────────────────────
def test_08_baseline_comparison():
    store.reset()
    suite = asyncio.run(
        SuiteRunner(simulation_mode=True).run_suite(
            mode="single", case_ids=["EVAL-001"], include_baseline=True,
            baseline_systems=["baseline_pipeline"],
        )
    )
    comp = suite["baseline_comparison"]
    assert "baseline_pipeline" in comp
    rows = {r["metric"]: r for r in comp["baseline_pipeline"]["rows"]}
    assert rows[M.GROUNDEDNESS]["available"] is True
    # Direction is tri-state and correct for lower-is-better metrics.
    assert rows[M.LATENCY]["higher_is_better"] is False
    for row in rows.values():
        if row["available"]:
            assert row["direction"] in {"better", "worse", "equal"}
        else:
            assert row["unavailable_reason"], "an unavailable metric must explain itself"


# ── 9. Human score storage ──────────────────────────────────
def test_09_human_score_storage():
    store.reset()
    rid = "ev-test-human"
    human.submit(HumanEvaluation(evaluation_run_id=rid, reviewer_id="r1",
                                 accuracy_score=4, overall_score=4, decision="PASS"))
    human.submit(HumanEvaluation(evaluation_run_id=rid, reviewer_id="r2",
                                 accuracy_score=2, overall_score=2, decision="FAIL",
                                 comment="weaker evidence"))
    agg = human.aggregate(rid)
    assert agg["available"] and agg["reviewer_count"] == 2
    assert agg["average_overall"] == 3.0
    assert agg["score_variance"] > 0            # disagreement is not hidden
    assert agg["decisions"] == {"PASS": 1, "FAIL": 1}
    assert agg["comments"] and agg["comments"][0]["reviewer_id"] == "r2"

    # Automated vs human surfaces the gap rather than blending it away.
    cmp = human.compare_with_automated(rid, {M.ACCURACY: {"available": True, "value": 1.0}})
    row = next(r for r in cmp["rows"] if r["metric"] == M.ACCURACY)
    assert row["gap"] is not None and cmp["disagreement_detected"] is True
    # A resubmission by the same reviewer replaces, not duplicates.
    human.submit(HumanEvaluation(evaluation_run_id=rid, reviewer_id="r1", overall_score=5))
    assert human.aggregate(rid)["reviewer_count"] == 2


# ── 10. Regression detection ────────────────────────────────
def test_10_regression_detection():
    store.reset()
    before = {
        M.GROUNDEDNESS: {"available": True, "value": 0.90, "unit": "ratio"},
        M.HALLUCINATION_RATE: {"available": True, "value": 0.10, "unit": "ratio"},
        M.LATENCY: {"available": True, "value": 1000.0, "unit": "ms"},
        "overall_score": 0.9,
    }
    store.save_suite({"suite_id": "s-old", "aggregate": before, "runs": [], "counts": {}})
    after = {
        M.GROUNDEDNESS: {"available": True, "value": 0.70, "unit": "ratio"},   # regressed
        M.HALLUCINATION_RATE: {"available": True, "value": 0.02, "unit": "ratio"},  # improved
        M.LATENCY: {"available": True, "value": 1050.0, "unit": "ms"},         # within noise
        "overall_score": 0.8,
    }
    reg = compare_with_previous("s-new", after)
    assert reg["compared"] is True
    directions = {c["metric"]: c["direction"] for c in reg["changes"]}
    assert directions[M.GROUNDEDNESS] == "regressed"
    assert directions[M.HALLUCINATION_RATE] == "improved"   # lower is better
    assert directions[M.LATENCY] == "unchanged"             # under the noise floor
    assert reg["regression_count"] == 1 and reg["overall_delta"] == -0.1


# ── 11. Empty evaluation suite ──────────────────────────────
def test_11_empty_suite_is_handled():
    store.reset()
    suite = asyncio.run(
        SuiteRunner(simulation_mode=True).run_suite(mode="single", case_ids=["NOPE-999"])
    )
    assert suite["status"] == "completed"
    assert suite["counts"]["cases"] == 0
    assert suite["aggregate"]["overall_score"] is None
    assert suite["notes"], "an empty suite explains itself instead of erroring"


# ── 12. Partial / failed agent run ──────────────────────────
def test_12_partial_or_failed_agent_run():
    # A run that produced nothing must not be scored as a success.
    broken = run(status="failed", findings=[], insights=[], summary="")
    case = simple_case(expected_subtasks=["produce prioritized insights"])
    eng = EvaluationEngine()
    results = eng.measure(case, broken)
    from app.evaluation.schemas import EvaluationRun
    ev = EvaluationRun(evaluation_run_id="x", case_id="T-1", case_name="t",
                       scenario_type="NORMAL")
    ev.metrics = {k: v.to_dict() for k, v in results.items()}
    outcome, reasons, gates = eng.decide_outcome(case, ev)
    assert outcome in {"PARTIAL", "FAIL"}
    assert gates and reasons


# ── 13. Missing ground truth ────────────────────────────────
def test_13_missing_ground_truth_is_unavailable_not_zero():
    from app.evaluation.automated import CorrectnessEvaluator
    a = CorrectnessEvaluator().evaluate(simple_case(), run())
    assert a.available is False
    assert "no checkable ground truth" in a.unavailable_reason
    assert a.value is None, "must never fabricate a score without ground truth"


# ── 14. Insufficient evidence ───────────────────────────────
def test_14_insufficient_evidence():
    e = EvidenceEvaluator().evaluate(simple_case(), run(findings=[]))
    assert e.available is False and e.unavailable_reason

    weak = run(findings=[finding(credibility="unverified", relevance=0.36,
                                 url="https://medium.com/x", published_date=None)])
    e2 = EvidenceEvaluator().evaluate(simple_case(), weak)
    assert e2.available and e2.value < 0.5
    assert e2.details["weak_source_count"] == 1
    assert e2.details["credible_source_count"] == 0


# ── 15. Unsupported conclusion ──────────────────────────────
def test_15_unsupported_conclusion():
    case = simple_case(expected_refusal=True)
    # Correct refusal → rate 0 (best) and counted as a success.
    refused = run(summary="There is no evidence to support this claim; it cannot be confirmed.")
    r = RefusalEvaluator().evaluate(case, refused)
    assert r.available and r.value == 0.0
    assert r.details["unsupported_conclusion_refusal"] is True

    # Asserting it anyway → rate 1 and a critical failure.
    asserted = run(summary="This confirms that the company built the system in secret.")
    r2 = RefusalEvaluator().evaluate(case, asserted)
    assert r2.value == 1.0

    # Echoing the user's own goal is not an assertion.
    echoed = run(
        goal="Prove that OpenAI secretly developed an unreleased system",
        summary='No evidence found relevant to "Prove that OpenAI secretly developed an '
                'unreleased system" — the claim cannot be confirmed.',
    )
    r3 = RefusalEvaluator().evaluate(case, echoed)
    assert r3.value == 0.0, "quoting the request must not be read as asserting it"

    # A case that does not ask for a conclusion is not scored on this metric.
    assert RefusalEvaluator().evaluate(simple_case(), refused).available is False


# ── 16. Contradictory evidence ──────────────────────────────
def test_16_contradictory_evidence():
    case = dataset.get_case("EVAL-008")
    assert case.expects_conflict_detection is True
    ev, raw = asyncio.run(EvaluationEngine().evaluate_case(case, simulation_mode=True))
    conflicts = (raw.get("framework") or {}).get("conflicting_evidence") or []
    assert conflicts, "the contradiction fixture must be detected"
    assert all(c.get("verdict") for c in conflicts), "each conflict is adjudicated"
    # Uncertainty must be represented, and the case must not silently pass by
    # picking one side without a verdict.
    u = ev.metrics[M.UNCERTAINTY_HANDLING]
    assert u["available"] and u["value"] >= 0.5
    assert ev.outcome in {"PASS", "PARTIAL"}


# ── 17. Tool failure ────────────────────────────────────────
def test_17_tool_failure():
    case = dataset.get_case("EVAL-009")
    ev, raw = asyncio.run(EvaluationEngine().evaluate_case(case, simulation_mode=True))
    fw = raw.get("framework") or {}
    assert fw.get("fallback_history"), "a failure was injected and a fallback engaged"
    rec = ev.metrics[M.RECOVERY_RATE]
    assert rec["available"] and rec["value"] == 1.0
    assert raw.get("status") in {"completed", "completed_partial"}
    assert ev.outcome in {"PASS", "PARTIAL"}


# ── extra: uncertainty calibration + robustness/reliability ──
def test_18_uncertainty_and_suite_aggregation():
    # Confident on weak evidence → fails the calibration check.
    over = UncertaintyEvaluator().evaluate(
        simple_case(), run(framework={"overall_confidence": 0.9}), 0.30
    )
    assert over.value == 0.0 and "weak evidence" in over.details["verdict"]
    # Correctly uncertain where the case requires it → passes.
    ok = UncertaintyEvaluator().evaluate(
        simple_case(expects_uncertainty=True),
        run(framework={"overall_confidence": 0.4, "uncertainty_flags": ["unresolved"]}), 0.30,
    )
    assert ok.value == 1.0

    rel = ReliabilityEvaluator().evaluate(
        ["PASS", "PASS", "FAIL"], ["completed"] * 3, [100.0, 120.0, 90.0]
    )
    # Metric values are rounded to 4dp for reporting, so compare at that precision.
    assert abs(rel.value - (2 / 3)) < 1e-4
    assert rel.details["failed_runs"] == 1, "failed repetitions are reported, not hidden"

    rob = RobustnessEvaluator().evaluate({
        "NORMAL": {"total": 2, "score": 1.0},
        "ADVERSARIAL": {"total": 1, "score": 0.0},
        "UNUSED": {"total": 0, "score": 0.0},
    })
    # Unweighted mean, so a strong category cannot mask the collapse.
    assert rob.value == 0.5
    assert rob.details["weakest_category"] == "ADVERSARIAL"


# ═════════════════════════════════════════════════════════════
# Regression tests for the Task 6 fix round
# ═════════════════════════════════════════════════════════════
def test_21_repeated_runs_group_by_stable_case_id():
    """Repetitions must share case_id and differ only by repeat_index.

    Regression: reliability/consistency depend on grouping by the stable case
    identifier. Grouping by evaluation_run_id (unique per execution) would put every
    repetition in its own bucket and make both metrics permanently unmeasurable.
    """
    store.reset()
    suite = asyncio.run(
        SuiteRunner(simulation_mode=True).run_repeated("EVAL-011", repeats=3)
    )
    mine = [r for r in suite["runs"] if r["system"] == "insightpulse"]
    assert len(mine) == 3
    assert {r["case_id"] for r in mine} == {"EVAL-011"}, "one stable case_id"
    assert sorted(r["repeat_index"] for r in mine) == [0, 1, 2]
    # Each execution keeps its own identity.
    assert len({r["evaluation_run_id"] for r in mine}) == 3
    assert len({r["agent_run_id"] for r in mine}) == 3


def test_22_consistency_is_measured_from_repeated_runs():
    """Regression: consistency was computed per case but never rolled into the
    suite aggregate, so the dashboard read it as unavailable."""
    store.reset()
    suite = asyncio.run(
        SuiteRunner(simulation_mode=True).run_repeated("EVAL-011", repeats=3)
    )
    entry = suite["aggregate"][M.CONSISTENCY]
    assert entry["available"] is True, "consistency must not be n/a with 3 repetitions"
    assert 0.0 <= entry["value"] <= 1.0
    assert entry["details"]["cases_measured"] == ["EVAL-011"]
    assert entry["details"]["repetitions"] == 3
    assert entry["details"]["pairs_compared"] == 3
    # Per-case block is retained alongside the rollup.
    assert suite["consistency"]["EVAL-011"]["available"] is True


def test_23_reliability_is_measured_from_repeated_runs():
    store.reset()
    suite = asyncio.run(
        SuiteRunner(simulation_mode=True).run_repeated("EVAL-011", repeats=3)
    )
    entry = suite["aggregate"][M.RELIABILITY]
    assert entry["available"] is True, "reliability must not be n/a with 3 repetitions"
    d = entry["details"]
    assert d["total_runs"] == 3
    assert d["successful_runs"] + d["partial_runs"] + d["failed_runs"] == 3
    # Formula is successful / total, not task completion.
    assert abs(entry["value"] - d["successful_runs"] / d["total_runs"]) < 1e-4

    # A suite with no repetition reports it as unmeasurable, not as zero.
    store.reset()
    single = asyncio.run(
        SuiteRunner(simulation_mode=True).run_suite(
            mode="single", case_ids=["EVAL-001"], include_baseline=False
        )
    )
    rel = single["aggregate"][M.RELIABILITY]
    assert rel["available"] is False and rel["value"] is None
    assert "repeated" in rel["unavailable_reason"]


def test_24_incomplete_scenario_is_executed_by_the_demo_suite():
    """Regression: the demo suite omitted the INCOMPLETE case, so that row of the
    scenario matrix showed 'not run'."""
    ids = [c.case_id for c in dataset.demo_suite()]
    scenarios = {c.scenario_type for c in dataset.demo_suite()}
    assert "EVAL-006" in ids
    # Every required scenario class is present in the demo suite itself.
    assert {"NORMAL", "AMBIGUOUS", "ADVERSARIAL", "CONTRADICTORY", "INCOMPLETE",
            "TOOL_FAILURE", "UNSUPPORTED_CONCLUSION"} <= scenarios

    case = dataset.get_case("EVAL-006")
    ev, raw = asyncio.run(EvaluationEngine().evaluate_case(case, simulation_mode=True))
    assert ev.status == "completed", "the incomplete case must actually execute"
    assert ev.outcome in {"PASS", "PARTIAL", "FAIL"}, "a real measured outcome"
    assert ev.agent_run_id, "measured against a real agent run"
    # It must not be marked passed without evidence: the subtask checks are recorded.
    checks = ev.metrics[M.TASK_COMPLETION]["details"]["checks"]
    assert any("missing subject" in c["subtask"] for c in checks)


def test_25_blocked_baseline_is_not_comparable():
    """A baseline that produced nothing must not yield a flattering comparison."""
    store.reset()
    suite = asyncio.run(
        SuiteRunner(simulation_mode=True).run_suite(
            mode="single", case_ids=["EVAL-001"], include_baseline=True,
            baseline_systems=["baseline_llm"],
        )
    )
    comp = suite["baseline_comparison"]["baseline_llm"]
    if comp["blocked"]:
        assert comp["blocked_reason"], "the real failure reason is recorded"
        assert all(r["direction"] == "not_comparable" for r in comp["rows"])
        assert all(r["available"] is False for r in comp["rows"])
        assert all(r["baseline"] is None for r in comp["rows"]), "no fake zero scores"
        assert all(r["difference"] is None for r in comp["rows"]), \
            "no percentage difference against an unavailable system"
    else:
        # Provider available: real values, and every row still declares direction.
        assert any(r["available"] for r in comp["rows"])


def test_26_lower_is_better_comparison_direction():
    """Direction must come from the metric's declared semantics, never the sign."""
    runner = SuiteRunner(simulation_mode=True)
    primary = [{
        "system": "insightpulse", "case_id": "C1",
        "metrics": {
            M.LATENCY: {"available": True, "value": 400.0},
            M.HALLUCINATION_RATE: {"available": True, "value": 0.05},
            M.GROUNDEDNESS: {"available": True, "value": 0.95},
        },
    }]
    baseline = [{
        "system": "baseline_pipeline", "case_id": "C1", "provenance": {},
        "metrics": {
            M.LATENCY: {"available": True, "value": 650.0},
            M.HALLUCINATION_RATE: {"available": True, "value": 0.20},
            M.GROUNDEDNESS: {"available": True, "value": 0.80},
        },
    }]
    rows = {
        r["metric"]: r
        for r in runner._baseline_comparison(primary, baseline)["baseline_pipeline"]["rows"]  # noqa: SLF001
    }
    # Lower latency than baseline is an improvement, even though the difference is negative.
    assert rows[M.LATENCY]["difference"] == -250.0
    assert rows[M.LATENCY]["direction"] == "better"
    assert rows[M.LATENCY]["higher_is_better"] is False
    # Lower hallucination is likewise better despite a negative difference.
    assert rows[M.HALLUCINATION_RATE]["difference"] < 0
    assert rows[M.HALLUCINATION_RATE]["direction"] == "better"
    # Higher groundedness is better with a positive difference.
    assert rows[M.GROUNDEDNESS]["difference"] > 0
    assert rows[M.GROUNDEDNESS]["direction"] == "better"

    # And the inverse: slower than baseline must read as worse.
    slower = [{
        "system": "insightpulse", "case_id": "C1",
        "metrics": {M.LATENCY: {"available": True, "value": 900.0}},
    }]
    row = next(
        r for r in runner._baseline_comparison(slower, baseline)["baseline_pipeline"]["rows"]  # noqa: SLF001
        if r["metric"] == M.LATENCY
    )
    assert row["difference"] > 0 and row["direction"] == "worse"


def test_27_human_review_persists_and_updates_the_queue():
    store.reset()
    suite = asyncio.run(
        SuiteRunner(simulation_mode=True).run_suite(
            mode="single", case_ids=["EVAL-001"], include_baseline=False
        )
    )
    rid = next(r["evaluation_run_id"] for r in suite["runs"] if r["system"] == "insightpulse")

    before = human.pending_and_completed(store.latest_suite())
    assert any(p["evaluation_run_id"] == rid for p in before["pending"])
    assert not before["completed"]

    human.submit(HumanEvaluation(
        evaluation_run_id=rid, reviewer_id="verifier",
        accuracy_score=4, completion_score=5, evidence_score=3,
        groundedness_score=4, uncertainty_score=4, actionability_score=4,
        overall_score=4, decision="PASS", comment="verified during the fix round",
    ))

    after = human.pending_and_completed(store.latest_suite())
    assert any(c["evaluation_run_id"] == rid for c in after["completed"])
    assert not any(p["evaluation_run_id"] == rid for p in after["pending"])
    assert len(after["completed"]) == len(before["completed"]) + 1
    assert len(after["pending"]) == len(before["pending"]) - 1

    agg = human.aggregate(rid)
    assert agg["reviewer_count"] == 1
    assert agg["comments"][0]["comment"] == "verified during the fix round"
    # The stored score survives a fresh read of the store.
    assert store.human_reviews(rid)[0]["overall_score"] == 4
    assert store.human_review_count() == 1


def test_28_scenario_coverage_aggregation():
    store.reset()
    suite = asyncio.run(
        SuiteRunner(simulation_mode=True).run_suite(mode="demo", include_baseline=False)
    )
    matrix = suite["scenario_matrix"]
    executed = {k for k, v in matrix.items() if v["total"]}
    # Every required class is exercised, INCOMPLETE included.
    assert executed == set(dataset.SCENARIO_TYPES), f"missing: {set(dataset.SCENARIO_TYPES) - executed}"
    for name, bucket in matrix.items():
        total = bucket["total"]
        assert bucket["passed"] + bucket["partial"] + bucket["failed"] + bucket["error"] == total
        if total:
            expected = (bucket["passed"] + 0.5 * bucket["partial"]) / total
            assert abs(bucket["score"] - expected) < 1e-4, f"{name} score mismatch"
    # Robustness is the unweighted mean, so a weak category is not hidden.
    rob = suite["aggregate"]["robustness"]
    assert rob["available"] is True
    scores = [v["score"] for v in matrix.values() if v["total"]]
    assert abs(rob["value"] - sum(scores) / len(scores)) < 1e-3


# ── extra: report export in every format ────────────────────
def test_20_report_export_formats():
    from app.evaluation.reports import (
        build_evaluation_report,
        render_evaluation_html,
        render_evaluation_markdown,
        render_evaluation_pdf,
    )

    store.reset()
    suite = asyncio.run(
        SuiteRunner(simulation_mode=True).run_suite(
            mode="single", case_ids=["EVAL-001", "EVAL-009"], include_baseline=True,
            baseline_systems=["baseline_pipeline"],
        )
    )
    report = build_evaluation_report(suite)
    # Structured payload carries the mandated sections.
    for key in ("executive_summary", "methodology", "scenario_coverage", "results",
                "case_results", "baseline_comparison", "reliability", "consistency",
                "human_review", "failures", "uncertainty_cases", "recovery_cases",
                "regression", "recommendations", "provenance"):
        assert key in report, f"evaluation report is missing '{key}'"

    md = render_evaluation_markdown(report)
    assert "# InsightPulse — Evaluation Report" in md
    assert "## Metric methodology" in md
    # Unavailable metrics are explained in the export, never printed as a number.
    assert "not comparable" in md or "not measurable" in md

    html = render_evaluation_html(report)
    assert html.startswith("<!DOCTYPE html>") and "Evaluation Report" in html

    pdf = render_evaluation_pdf(report)
    assert pdf[:4] == b"%PDF" and len(pdf) > 3000


# ── extra: the full demo suite runs end to end ──────────────
def test_19_demo_suite_end_to_end():
    store.reset()
    suite = asyncio.run(
        SuiteRunner(simulation_mode=True).run_suite(mode="demo", include_baseline=False)
    )
    assert suite["status"] == "completed"
    assert suite["counts"]["runs"] >= 7
    matrix = suite["scenario_matrix"]
    executed = {k for k, v in matrix.items() if v["total"]}
    # Every required scenario class is exercised by the demo suite.
    assert {"NORMAL", "AMBIGUOUS", "CONTRADICTORY", "TOOL_FAILURE",
            "UNSUPPORTED_CONCLUSION", "ADVERSARIAL"} <= executed
    assert suite["reliability"] and suite["consistency"]
    assert isinstance(suite["aggregate"]["overall_score"], float)
    # Thresholds used for the verdicts are recorded with the suite.
    assert suite["thresholds"]["min_groundedness"] == Thresholds().min_groundedness
