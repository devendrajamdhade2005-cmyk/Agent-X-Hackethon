"""Task 7 — tracing, diagnosis and verified self-improvement.

The 30 checks from the brief (section 47).

Structure mirrors the rest of the suite: pure logic (redaction, policy bounds,
comparison verdicts, analyzer arithmetic) is tested directly against hand-built
fixtures so it is fast and deterministic, while trace integrity, controlled failure
and the full improvement loop drive the *real* runtime in simulation mode.

Nothing here mocks the thing under test. The controlled failure genuinely raises
inside the production retry loop, the retries genuinely sleep, the improvement
genuinely changes runtime policy, and the re-run genuinely happens.

Two expensive artefacts — one traced run and one full improvement cycle — are built
once and shared, because each drives the whole agent graph. Module-global stores are
reset in `setup_function` since they are shared across the process, the same
convention the rest of the suite uses.
"""

from __future__ import annotations

import asyncio

from app.observability import controlled_failure as cf
from app.observability import policy as policy_mod
from app.observability import redaction
from app.observability.analyzer import RootCauseAnalyzer, TraceAnalyzer
from app.observability.improvement import (
    MAX_QUALITY_DROP,
    METRIC_DIRECTION,
    MIN_LATENCY_GAIN_MS,
    ComparisonEngine,
    ImprovementEngine,
)
from app.observability.loop import SelfImprovementLoop
from app.observability.policy import BOUNDS, PolicyRegistry
from app.observability.providers import ExternalTraceProvider, local_provider
from app.observability.schemas import ERROR_CATEGORIES, ROOT_CAUSES, SPAN_KINDS
from app.observability.tracer import get_tracer, reset_span_stack
from app.sources.resilience import registry as resilience_registry

GOAL = "Track advances in AI agent architectures"
KEYWORDS = ["AI agents"]

_traced_run: dict | None = None
_cycle: dict | None = None


def setup_function() -> None:
    """Module-global state is shared across the process, so reset what a test owns."""
    policy_mod.registry.reset()
    cf.controller.reset()
    resilience_registry.reset()
    reset_span_stack()


# ── shared fixtures (expensive: each drives the real graph) ──
def traced_run() -> tuple[dict, dict]:
    """One real traced run with a controlled failure armed. Built once."""
    global _traced_run
    if _traced_run is None:
        from app.graph.runner import run_graph

        run_id = "t7testrun001"
        cf.controller.arm(run_id=run_id, target_source="semantic_scholar",
                          failure_type="rate_limit", failure_count=2)
        result = asyncio.run(run_graph(
            GOAL, keywords=KEYWORDS, simulation_mode=True,
            scenario="test_scenario", run_id=run_id,
        ))
        cf.controller.disarm(run_id)
        trace = local_provider.get(str(result.get("trace_id") or "")) or {}
        _traced_run = {"result": result, "trace": trace}
    return _traced_run["result"], _traced_run["trace"]


def cycle_report() -> dict:
    """One real end-to-end improvement cycle. Built once."""
    global _cycle
    if _cycle is None:
        policy_mod.registry.reset()
        _cycle = asyncio.run(SelfImprovementLoop().execute(
            target_source="semantic_scholar", failure_type="rate_limit",
            failure_count=2, simulation_mode=True, validate_with_evaluation=True,
        ))
    return _cycle


def spans_by_kind(trace: dict, kind: str) -> list[dict]:
    return [s for s in (trace.get("spans") or []) if s.get("kind") == kind]


# ═══════════════════════════════════════════════════════════
# TRACE INTEGRITY & SPAN HIERARCHY (1–8)
# ═══════════════════════════════════════════════════════════
def test_01_every_run_produces_a_trace_with_one_root_span():
    result, trace = traced_run()
    assert result.get("trace_id"), "the run result must carry its trace id"
    assert trace, "the trace must be retrievable from the provider"
    roots = [s for s in trace["spans"] if not s.get("parent_span_id")]
    assert len(roots) == 1, f"expected exactly one root span, got {len(roots)}"
    assert roots[0]["kind"] == "run"


def test_02_parent_child_integrity_no_orphan_spans():
    _, trace = traced_run()
    ids = {s["span_id"] for s in trace["spans"]}
    orphans = [
        s["name"] for s in trace["spans"]
        if s.get("parent_span_id") and s["parent_span_id"] not in ids
    ]
    assert not orphans, f"every child must reference a recorded parent; orphans: {orphans}"


def test_03_span_kinds_cover_the_whole_execution():
    _, trace = traced_run()
    kinds = {s["kind"] for s in trace["spans"]}
    # The point of the trace is that it explains the run end to end, so each layer
    # of the architecture has to be represented.
    for required in ("run", "node", "agent", "tool", "provider", "decision"):
        assert required in kinds, f"no {required} span was recorded"
    assert kinds <= set(SPAN_KINDS), f"unknown span kind: {kinds - set(SPAN_KINDS)}"


def test_04_agent_spans_own_their_tool_spans():
    _, trace = traced_run()
    by_id = {s["span_id"]: s for s in trace["spans"]}
    agents = spans_by_kind(trace, "agent")
    tools = spans_by_kind(trace, "tool")
    assert agents and tools

    def ancestors(span):
        out, parent = [], span.get("parent_span_id")
        while parent and parent in by_id:
            out.append(by_id[parent])
            parent = by_id[parent].get("parent_span_id")
        return out

    for tool in tools:
        chain = ancestors(tool)
        assert any(a["kind"] == "agent" for a in chain), (
            f"tool span {tool['name']} is not nested under an agent — parallel "
            f"branches must not share a parent by accident"
        )


def test_05_tool_spans_own_their_provider_spans():
    _, trace = traced_run()
    by_id = {s["span_id"]: s for s in trace["spans"]}
    providers = spans_by_kind(trace, "provider")
    assert providers
    for prov in providers:
        parent = by_id.get(str(prov.get("parent_span_id") or ""))
        assert parent is not None and parent["kind"] == "tool", (
            f"provider span {prov['name']} must hang off the tool that called it"
        )


def test_06_every_span_has_a_status_and_sane_duration():
    _, trace = traced_run()
    for span in trace["spans"]:
        assert span.get("status") in {"ok", "error", "degraded"}, span
        assert isinstance(span.get("duration_ms"), int)
        assert span["duration_ms"] >= 0, f"negative duration on {span['name']}"


def test_07_trace_carries_correlation_ids():
    result, trace = traced_run()
    assert trace["run_id"] == result["run_id"], "trace must be joinable to the run"
    assert trace.get("scenario") == "test_scenario"
    for span in trace["spans"]:
        assert span.get("trace_id") == trace["trace_id"]


def test_08_latency_is_recorded_and_root_covers_the_children():
    _, trace = traced_run()
    root = [s for s in trace["spans"] if not s.get("parent_span_id")][0]
    assert trace["duration_ms"] > 0
    deepest = max(s["duration_ms"] for s in trace["spans"])
    # The root span brackets the whole run, so nothing inside it can be longer.
    assert root["duration_ms"] >= deepest, "a child outlasted the run span"


# ═══════════════════════════════════════════════════════════
# PROMPT, DECISION & TOKEN ACCOUNTING (9–12)
# ═══════════════════════════════════════════════════════════
def test_09_llm_spans_record_prompt_shape_but_never_prompt_text():
    _, trace = traced_run()
    llm_spans = spans_by_kind(trace, "llm")
    assert llm_spans, "an LLM span should be recorded even when the call fails"
    for span in llm_spans:
        attrs = span["attributes"]
        assert "prompt_type" in attrs
        assert isinstance(attrs.get("system_prompt_chars"), int)
        assert isinstance(attrs.get("user_prompt_chars"), int)
        # Only the shape of the prompt is observable, never its content.
        for key in ("system", "user", "prompt", "messages", "completion"):
            assert key not in attrs, f"prompt content leaked into the span via '{key}'"


def test_10_unmeasured_tokens_are_reported_unavailable_not_zero():
    _, trace = traced_run()
    tokens = trace["token_usage"]
    assert tokens["status"] in {"measured", "unavailable"}
    if tokens["status"] == "unavailable":
        # A provider that reported nothing must not be recorded as "0 tokens used";
        # that would be a fabricated measurement.
        assert tokens["reason"], "an unavailable metric must carry its reason"
        assert tokens.get("estimated_cost_usd") is None


def test_11_measured_tokens_are_recorded_when_the_provider_reports_them():
    tracer = get_tracer(run_id="tok-test", goal="t", scenario="unit")
    span = tracer.start_span("llm:probe", "llm")
    tracer.record_tokens(input_tokens=120, output_tokens=45, model="test-model",
                         span_id=span, prompt_type="probe")
    tracer.end_span(span, status="ok")
    trace = tracer.finish(status="ok").to_dict()
    tokens = trace["token_usage"]
    assert tokens["status"] == "measured"
    assert tokens["input_tokens"] == 120
    assert tokens["output_tokens"] == 45
    assert tokens["total_tokens"] == 165

    # And the zero case is explicitly *not* a measurement.
    reset_span_stack()
    t2 = get_tracer(run_id="tok-zero", goal="t", scenario="unit")
    t2.record_tokens(input_tokens=0, output_tokens=0, model="test-model")
    assert t2.finish(status="ok").to_dict()["token_usage"]["status"] == "unavailable"


def test_12_decision_spans_record_the_branch_taken():
    _, trace = traced_run()
    decisions = spans_by_kind(trace, "decision")
    assert decisions, "routing decisions must be traced"
    decided = [
        s for s in decisions
        if s["attributes"].get("decision")
        or any(e["name"] == "decision_made" for e in (s.get("events") or []))
    ]
    assert decided, "at least one decision span must record the route it chose"


# ═══════════════════════════════════════════════════════════
# ERRORS, RETRIES & RECOVERY (13–16)
# ═══════════════════════════════════════════════════════════
def test_13_errors_are_categorised_with_their_component():
    _, trace = traced_run()
    errors = trace["errors"]
    assert errors, "the controlled failure must appear in the trace"
    for err in errors:
        assert err["error_type"] in ERROR_CATEGORIES, err["error_type"]
        assert err["component"], "an error must say what failed"
        assert err.get("error_id") and err.get("trace_id") == trace["trace_id"]


def test_14_injected_errors_are_labelled_as_injected():
    _, trace = traced_run()
    injected = [e for e in trace["errors"] if e.get("injected")]
    assert len(injected) == 2, f"expected 2 injected failures, got {len(injected)}"
    for err in injected:
        assert err["error_type"] == "RATE_LIMIT"
        assert err["http_status"] == 429
        assert err["retryable"] is True
        assert err["provider"] == "semantic_scholar"


def test_15_retry_attempts_and_backoff_are_measured_on_the_provider_span():
    _, trace = traced_run()
    failed = [
        s for s in spans_by_kind(trace, "provider")
        if s["attributes"].get("provider") == "semantic_scholar"
        and s["status"] == "error"
    ]
    assert failed, "the rate-limited provider call must be recorded as failed"
    span = failed[0]
    assert span["attributes"]["attempts"] == 2, "the retry loop really ran twice"
    # The backoff is real wall-clock time, so it must be measured, not assumed.
    assert span["attributes"]["retry_wait_ms"] > 0
    assert any(e["name"] == "retry_recorded" for e in span["events"])


def test_16_model_failure_is_recorded_as_recovered_via_the_heuristic_path():
    _, trace = traced_run()
    model_errors = [e for e in trace["errors"] if e["error_type"] == "MODEL_ERROR"]
    if not model_errors:
        return  # a healthy model quota means there is nothing to assert here
    for err in model_errors:
        # Every LLM caller has a deterministic fallback, so the run does recover.
        assert err["recovery_status"] == "recovered"
        assert err["fallback_attempted"] is True


# ═══════════════════════════════════════════════════════════
# CONTROLLED FAILURE (17–20)
# ═══════════════════════════════════════════════════════════
def test_17_controlled_failure_is_deterministic():
    from app.graph.runner import run_graph

    counts = []
    for i in range(2):
        policy_mod.registry.reset()
        resilience_registry.reset()
        run_id = f"determinism-{i}"
        cf.controller.arm(run_id=run_id, target_source="semantic_scholar",
                          failure_type="rate_limit", failure_count=2)
        result = asyncio.run(run_graph(
            GOAL, keywords=KEYWORDS, simulation_mode=True,
            scenario="determinism", run_id=run_id,
        ))
        cf.controller.disarm(run_id)
        trace = local_provider.get(str(result["trace_id"])) or {}
        counts.append(sum(1 for e in trace["errors"] if e.get("injected")))
    assert counts[0] == counts[1] == 2, f"injection was not deterministic: {counts}"


def test_18_injection_is_scoped_to_one_run_id():
    from app.graph.runner import run_graph

    cf.controller.arm(run_id="armed-run", target_source="semantic_scholar",
                      failure_type="rate_limit", failure_count=2)
    # A different run in the same process must not see the armed failure.
    other = asyncio.run(run_graph(
        GOAL, keywords=KEYWORDS, simulation_mode=True,
        scenario="unarmed", run_id="different-run",
    ))
    trace = local_provider.get(str(other["trace_id"])) or {}
    assert not [e for e in trace["errors"] if e.get("injected")], (
        "an injection armed for one run leaked into another"
    )
    assert other["status"] == "completed"


def test_19_injection_drives_the_real_retry_loop():
    """The failure must be a real exception in the production path, not a stub."""
    import httpx

    from app.sources.registry import registry as sources
    from app.sources.resilience import collect_from_source

    async def scenario():
        cf.controller.arm(run_id="retry-loop", target_source="semantic_scholar",
                          failure_type="rate_limit", failure_count=2)
        cf.set_current_run("retry-loop")
        connector = sources.get("semantic_scholar")
        async with httpx.AsyncClient(timeout=5) as client:
            return await collect_from_source(
                connector, client, "quantum computing", simulation_mode=True)

    outcome = asyncio.run(scenario())
    cf.set_current_run("")
    assert outcome.attempts == 2, "the retry loop did not re-attempt"
    assert outcome.ok is False, "the source should be lost after its attempts"
    assert outcome.retry_wait_ms > 0, "the real backoff sleep was not measured"


def test_20_only_registered_sources_can_be_targeted():
    targets = cf.available_targets()
    assert "semantic_scholar" in targets
    assert "not_a_real_provider" not in targets
    # The API layer refuses anything outside this list, so a controlled failure can
    # never describe a provider the system does not actually call.
    assert all(isinstance(t, str) and t for t in targets)
    assert set(cf.FAILURE_TYPES) == {"rate_limit", "timeout", "server_error", "bad_response"}


# ═══════════════════════════════════════════════════════════
# ANALYZER & ROOT CAUSE (21–25)
# ═══════════════════════════════════════════════════════════
def test_21_analyzer_counts_match_the_trace():
    _, trace = traced_run()
    analysis = TraceAnalyzer().analyze(trace)
    assert analysis["span_count"] == len(trace["spans"])
    assert analysis["counts"]["errors"] == len(trace["errors"])
    assert analysis["counts"]["provider_calls"] == len(spans_by_kind(trace, "provider"))
    assert analysis["counts"]["agents"] == len(spans_by_kind(trace, "agent"))
    assert analysis["counts"]["injected_errors"] == 2


def test_22_wasted_retries_are_counted_per_call_not_summed_across_calls():
    """A provider called three times, failing once, wasted one retry — not three."""
    trace = {
        "trace_id": "t", "duration_ms": 1000, "status": "ok", "errors": [],
        "token_usage": {"status": "unavailable"}, "metrics": {},
        "spans": [
            {"span_id": "1", "kind": "provider", "name": "provider:x", "status": "error",
             "duration_ms": 500, "events": [], "attributes": {
                 "provider": "x", "attempts": 2, "latency_ms": 500, "retry_wait_ms": 400}},
            {"span_id": "2", "kind": "provider", "name": "provider:x", "status": "ok",
             "duration_ms": 5, "events": [], "attributes": {
                 "provider": "x", "attempts": 1, "latency_ms": 5, "retry_wait_ms": 0}},
            {"span_id": "3", "kind": "provider", "name": "provider:x", "status": "ok",
             "duration_ms": 5, "events": [], "attributes": {
                 "provider": "x", "attempts": 1, "latency_ms": 5, "retry_wait_ms": 0}},
        ],
    }
    waste = TraceAnalyzer().analyze(trace)["wasted_retries"]
    assert len(waste) == 1
    assert waste[0]["wasted_attempts"] == 1, (
        "attempts from separate calls are not retries of each other"
    )
    assert waste[0]["failed_calls"] == 1
    assert waste[0]["retry_wait_ms"] == 400


def test_23_diagnosis_identifies_excessive_retry_with_evidence():
    _, trace = traced_run()
    diagnosis = RootCauseAnalyzer().diagnose(trace)
    assert diagnosis.root_cause_type in ROOT_CAUSES
    assert diagnosis.root_cause_type in {"EXCESSIVE_RETRY", "RATE_LIMIT",
                                         "MULTIPLE_POSSIBLE_CAUSES"}
    assert diagnosis.affected_component == "semantic_scholar"
    assert 0.0 < diagnosis.confidence <= 1.0
    assert len(diagnosis.evidence) >= 2, "a conclusion needs its supporting evidence"
    assert any("429" in e or "rate-limit" in e for e in diagnosis.evidence)


def test_24_a_clean_trace_yields_unknown_rather_than_an_invented_cause():
    clean = {
        "trace_id": "clean", "duration_ms": 120, "status": "ok", "errors": [],
        "token_usage": {"status": "unavailable"}, "metrics": {},
        "spans": [
            {"span_id": "1", "kind": "run", "name": "agent_run", "status": "ok",
             "duration_ms": 120, "events": [], "attributes": {}},
            {"span_id": "2", "kind": "tool", "name": "research_search", "status": "ok",
             "parent_span_id": "1", "duration_ms": 10, "events": [],
             "attributes": {"tool": "research_search", "query": "a"}},
        ],
    }
    diagnosis = RootCauseAnalyzer().diagnose(clean)
    assert diagnosis.root_cause_type == "UNKNOWN"
    assert diagnosis.confidence == 0.0
    assert diagnosis.uncertain is True
    assert not diagnosis.improvement_type, "nothing to fix means nothing proposed"


def test_25_impact_never_fabricates_token_or_cost_figures():
    _, trace = traced_run()
    impact = RootCauseAnalyzer().diagnose(trace).impact
    if trace["token_usage"]["status"] != "measured":
        assert impact["token_overhead"] is None
        assert impact["token_overhead_note"]
        assert impact["estimated_cost_change_usd"] is None
        assert impact["estimated_cost_note"]
    # Latency attributed to the fault is the measured backoff, never a guess.
    assert isinstance(impact["latency_added_ms"], int)
    assert impact["latency_added_ms"] >= 0


# ═══════════════════════════════════════════════════════════
# IMPROVEMENT ENGINE (26–28)
# ═══════════════════════════════════════════════════════════
def test_26_improvement_is_versioned_bounded_and_reversible():
    _, trace = traced_run()
    engine = ImprovementEngine()
    diagnosis = RootCauseAnalyzer().diagnose(trace)
    plan = engine.propose(diagnosis)
    assert plan.improvement_type == "RETRY_POLICY"
    assert plan.changed_parameter == "retry_attempts[semantic_scholar]"
    assert plan.status == "proposed"

    before = policy_mod.registry.version
    engine.apply(plan)
    assert plan.status == "applied"
    assert policy_mod.registry.version == before + 1
    assert policy_mod.registry.active.retry_attempts_for("semantic_scholar", 3) == 1

    engine.revert(plan)
    assert plan.status == "reverted"
    assert policy_mod.registry.version == before
    assert policy_mod.registry.active.retry_attempts_for("semantic_scholar", 3) == 3


def test_27_policy_values_are_clamped_to_declared_bounds():
    reg = PolicyRegistry()
    lo, hi = BOUNDS["retry_attempts"]
    reg.apply(retry_attempts_by_source={"arxiv": 99}, reason="over")
    assert reg.active.retry_attempts_for("arxiv", 3) == hi
    reg.apply(retry_attempts_by_source={"arxiv": -5}, reason="under")
    assert reg.active.retry_attempts_for("arxiv", 3) == lo

    t_lo, t_hi = BOUNDS["timeout_seconds"]
    reg.apply(timeout_by_source={"arxiv": 999.0}, reason="over")
    assert reg.active.timeout_for("arxiv", 12.0) == t_hi


def test_28_the_engine_changes_configuration_not_source_code():
    """The improvement path must never write to a source file."""
    from pathlib import Path

    watched = [
        Path("app/sources/resilience.py"),
        Path("app/graph/runner.py"),
        Path("app/observability/policy.py"),
    ]
    before = {p: p.read_bytes() for p in watched if p.exists()}
    assert before, "expected to be running from the backend directory"

    _, trace = traced_run()
    engine = ImprovementEngine()
    plan = engine.propose(RootCauseAnalyzer().diagnose(trace))
    engine.apply(plan)
    engine.revert(plan)

    for path, original in before.items():
        assert path.read_bytes() == original, f"{path} was modified by the engine"


# ═══════════════════════════════════════════════════════════
# BEFORE / AFTER ACCEPTANCE (29–32)
# ═══════════════════════════════════════════════════════════
def test_29_metric_direction_is_declared_not_inferred():
    assert METRIC_DIRECTION["duration_ms"] is False, "lower latency is better"
    assert METRIC_DIRECTION["groundedness"] is True, "higher groundedness is better"
    assert METRIC_DIRECTION["hallucination_rate"] is False
    assert METRIC_DIRECTION["findings"] is True

    rows = ComparisonEngine().compare(
        before={"duration_ms": 1000, "findings": 10},
        after={"duration_ms": 400, "findings": 12},
        primary_metric="duration_ms",
    )["rows"]
    by_metric = {r["metric"]: r for r in rows}
    assert by_metric["duration_ms"]["direction"] == "improved"
    assert by_metric["findings"]["direction"] == "improved"


def test_30_quality_regression_is_rejected_even_when_latency_improves():
    verdict = ComparisonEngine().compare(
        before={"duration_ms": 1500, "groundedness": 0.95, "task_completion": 1.0},
        after={"duration_ms": 300, "groundedness": 0.55, "task_completion": 1.0},
        primary_metric="duration_ms",
    )
    assert verdict["improvement_verified"] is False
    assert verdict["verdict"] == "IMPROVEMENT_REJECTED"
    assert verdict["quality_regressions"], "the regression must be named"
    assert any("groundedness" in r for r in verdict["quality_regressions"])


def test_31_noise_level_change_is_not_called_an_improvement():
    verdict = ComparisonEngine().compare(
        before={"duration_ms": 1000, "groundedness": 0.9},
        after={"duration_ms": 1000 - (MIN_LATENCY_GAIN_MS - 10), "groundedness": 0.9},
        primary_metric="duration_ms",
    )
    assert verdict["verdict"] == "NO_MATERIAL_CHANGE"
    assert verdict["improvement_verified"] is False

    # An unmeasurable target metric is reported as such, not silently passed.
    missing = ComparisonEngine().compare(
        before={"groundedness": 0.9}, after={"groundedness": 0.9},
        primary_metric="duration_ms",
    )
    assert missing["verdict"] == "NOT_MEASURABLE"


def test_32_rising_errors_or_falling_success_block_acceptance():
    worse_errors = ComparisonEngine().compare(
        before={"duration_ms": 1500, "errors": 1},
        after={"duration_ms": 200, "errors": 6},
        primary_metric="duration_ms",
    )
    assert worse_errors["verdict"] == "IMPROVEMENT_REJECTED"

    worse_success = ComparisonEngine().compare(
        before={"duration_ms": 1500, "task_success": 1.0},
        after={"duration_ms": 200, "task_success": 0.0},
        primary_metric="duration_ms",
    )
    assert worse_success["verdict"] == "IMPROVEMENT_REJECTED"
    assert MAX_QUALITY_DROP > 0


# ═══════════════════════════════════════════════════════════
# THE FULL LOOP (33–35)
# ═══════════════════════════════════════════════════════════
def test_33_the_loop_runs_every_stage_in_order():
    report = cycle_report()
    stages = [s["stage"] for s in report["stages"]]
    assert stages == ["trace", "understand", "diagnose", "choose", "apply",
                      "rerun", "measure", "verify"], stages
    assert report["before_trace_id"] and report["after_trace_id"]
    assert report["before_trace_id"] != report["after_trace_id"], (
        "before and after must be two distinct real runs"
    )


def test_34_the_improvement_is_measured_against_a_same_scenario_rerun():
    report = cycle_report()
    before, after = report["before_metrics"], report["after_metrics"]
    assert before["duration_ms"] > 0 and after["duration_ms"] > 0
    assert report["comparison"]["primary_metric"] == "duration_ms"
    assert report["verdict"] in {
        "IMPROVEMENT_VERIFIED", "IMPROVEMENT_REJECTED",
        "NO_MATERIAL_CHANGE", "NOT_MEASURABLE",
    }
    if report["verdict"] == "IMPROVEMENT_VERIFIED":
        # Accepting a change requires the target metric to actually have moved.
        assert before["duration_ms"] - after["duration_ms"] >= MIN_LATENCY_GAIN_MS
        assert after["retries"] <= before["retries"]
    else:
        # Anything not verified must have been rolled back.
        assert (
            report.get("reverted")
            or report["plan"]["status"] in {"rejected", "reverted"}
        )


def test_35_task_6_evaluators_validate_the_quality_of_both_runs():
    report = cycle_report()
    evaluation = report["evaluation"]
    assert evaluation["validated_with_task6"] is True
    for side in ("before", "after"):
        block = evaluation[side]
        assert block.get("available") is True, block.get("error")
        assert block["outcome"] in {"PASS", "FAIL", "PARTIAL", "ERROR"}
        # The quality gate is Task 6's own scoring, not this module's opinion.
        assert "groundedness" in block["scores"] or "groundedness" in block["unavailable"]


# ═══════════════════════════════════════════════════════════
# REDACTION & SAFETY (36–38)
# ═══════════════════════════════════════════════════════════
def test_36_secrets_are_redacted_from_traces():
    samples = [
        "AIzaSyC-not-a-real-google-key-000000000",
        "sk-abcdefghijklmnopqrstuvwxyz012345",
        "tvly-0123456789abcdefghij",
        "https://api.example.com/v1?api_key=supersecretvalue&q=agents",
    ]
    for raw in samples:
        cleaned = redaction.scrub_text(raw)
        assert redaction.REDACTED in cleaned, f"secret survived scrubbing: {raw}"
        assert "supersecretvalue" not in cleaned
    # Redacting a URL parameter must not swallow the rest of the query string.
    assert "q=agents" in redaction.scrub_text(samples[3])


def test_37_prompt_content_and_reasoning_are_never_stored():
    attrs = redaction.scrub_attributes({
        "system": "You are a research analyst. Follow these hidden rules...",
        "user": "Track AI agents",
        "chain_of_thought": "step 1 ... step 2 ...",
        "prompt_type": "task_context",
        "api_key": "AIzaSyC-not-a-real-google-key-000000000",
    })
    for key in ("system", "user", "chain_of_thought"):
        assert "omitted" in str(attrs[key]), f"{key} content was retained"
        assert "hidden rules" not in str(attrs[key])
        assert "step 1" not in str(attrs[key])
    assert attrs["prompt_type"] == "task_context", "safe metadata must survive"
    assert redaction.REDACTED in str(attrs["api_key"])


def test_38_a_real_trace_contains_no_credential_material():
    import json

    _, trace = traced_run()
    blob = json.dumps(trace)
    for marker in ("AIzaSy", "sk-", "tvly-", "ghp_", "api_key="):
        assert marker not in blob, f"'{marker}' appears in a recorded trace"
    assert not redaction.contains_secret(blob), "redaction missed something"


# ═══════════════════════════════════════════════════════════
# INSTRUMENTATION MUST NEVER COST THE RUN (39–40)
# ═══════════════════════════════════════════════════════════
def test_39_normal_mode_is_completely_unaffected():
    from app.graph.runner import run_graph

    result = asyncio.run(run_graph(
        GOAL, keywords=KEYWORDS, simulation_mode=True, scenario="normal"))
    assert result["status"] == "completed"
    assert result["metrics"]["findings_total"] > 0
    assert result["metrics"]["tool_calls"] > 0
    trace = local_provider.get(str(result["trace_id"])) or {}
    assert not [e for e in trace["errors"] if e.get("injected")], (
        "no failure was armed, so none may appear"
    )
    assert trace["optimization_version"] == 0, "an unimproved run runs the defaults"


def test_41_the_classic_agent_path_is_traced_too():
    """The dashboard's scan button uses the pre-LangGraph loop, so it needs a trace."""
    from app.agents.agent import AgentRunRequest, InsightPulseAgent

    agent = InsightPulseAgent(simulation_mode=True)
    result = asyncio.run(agent.run(AgentRunRequest(goal=GOAL, keywords=KEYWORDS)))
    data = result.to_dict()
    assert data["status"] == "completed"

    trace = local_provider.by_run(data["run_id"])
    assert trace is not None, "a classic run must be traced like a graph run"
    roots = [s for s in trace["spans"] if not s.get("parent_span_id")]
    assert len(roots) == 1 and roots[0]["kind"] == "run"
    assert roots[0]["attributes"].get("runtime") == "classic"
    # Tools and their providers must still nest correctly on this path.
    assert spans_by_kind(trace, "tool"), "no tool span on the classic path"
    assert spans_by_kind(trace, "provider"), "no provider span on the classic path"
    ids = {s["span_id"] for s in trace["spans"]}
    assert not [
        s for s in trace["spans"]
        if s.get("parent_span_id") and s["parent_span_id"] not in ids
    ]


def test_40_export_failure_and_disabled_tracing_both_degrade_safely():
    # An unreachable external collector must not affect the run or the local trace.
    external = ExternalTraceProvider(
        endpoint="http://127.0.0.1:9/never-listening", api_key="secret-key-value",
        project="test",
    )
    external.record({"trace_id": "x", "spans": [], "errors": []})
    status = external.status()
    assert "secret-key-value" not in str(status), "the export key must never be exposed"

    # Tracing switched off: the run still completes and simply is not traced.
    tracer = get_tracer(run_id="off", goal="g", scenario="unit")
    tracer.enabled = False
    span = tracer.start_span("noop", "run")
    tracer.end_span(span, status="ok")
    assert tracer.finish(status="ok") is not None
    assert tracer.instrumentation_failures == 0
