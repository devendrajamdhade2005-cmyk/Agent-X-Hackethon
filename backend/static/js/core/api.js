/* Backend client. Endpoints and payload shapes are exactly the existing ones —
   this redesign changes presentation, not the API contract.

   Every request is built with apiUrl() so the backend host is configured in
   exactly one place (core/config.js): same-origin for local dev, the Render
   backend in production. */

import { apiUrl } from "./config.js";

async function readError(res) {
  try {
    const body = await res.json();
    return body.detail || body.message || `HTTP ${res.status}`;
  } catch {
    return `HTTP ${res.status}`;
  }
}

export async function getTools() {
  const res = await fetch(apiUrl("/api/agent/tools"));
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function getHealth() {
  const res = await fetch(apiUrl("/health"));
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

/**
 * Stream an agent run over SSE.
 * onEvent receives every decoded event: run_started | activity | result | error | done
 * Returns the final result object.
 */
export async function runAgentStream(payload, onEvent, signal) {
  const res = await fetch(apiUrl("/api/agent/run/stream"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(await readError(res));

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      let event;
      try { event = JSON.parse(line.slice(5).trim()); } catch { continue; }
      if (event.type === "result") result = event.result;
      if (event.type === "error") throw new Error(event.message || "agent error");
      onEvent(event);
    }
  }
  if (!result) throw new Error("the run ended without producing a report");
  return result;
}

/** Non-streaming fallback for environments where SSE is proxied away. */
export async function runAgent(payload) {
  const res = await fetch(apiUrl("/api/agent/run"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

/* ── LangGraph agent framework (Task 5) ─────────────────────── */
export async function getGraphInfo() {
  const res = await fetch(apiUrl("/api/agent/graph/info"));
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

/**
 * Stream a LangGraph run over SSE. `path` is the endpoint suffix:
 *   "/api/agent/graph/run/stream"  — normal run
 *   "/api/agent/graph/adversarial" — adversarial demo
 * onEvent receives every decoded event: run_started | activity | result | error | done
 */
export async function runGraphStream(path, payload, onEvent, signal) {
  const res = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(await readError(res));

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      let event;
      try { event = JSON.parse(line.slice(5).trim()); } catch { continue; }
      if (event.type === "result") result = event.result;
      if (event.type === "error") throw new Error(event.message || "graph error");
      onEvent(event);
    }
  }
  if (!result) throw new Error("the graph run ended without a result");
  return result;
}

/* ── Evaluation & benchmarking (Task 6) ─────────────────────── */
async function getJSON(path) {
  const res = await fetch(apiUrl(path));
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export const getEvaluationCases   = () => getJSON("/api/evaluation/cases");
export const getEvaluationMetrics = () => getJSON("/api/evaluation/metrics");
export const getEvaluationRuns    = () => getJSON("/api/evaluation/runs");
export const getEvaluationBaseline = () => getJSON("/api/evaluation/baseline");
export const getEvaluationHistory = () => getJSON("/api/evaluation/history");
export const getEvaluationHuman   = () => getJSON("/api/evaluation/human");
export const getEvaluationRun = (id) =>
  getJSON(`/api/evaluation/runs/${encodeURIComponent(id)}`);

export function evaluationReportUrl(format = "md", suiteId = "") {
  const q = new URLSearchParams({ format });
  if (suiteId) q.set("suite_id", suiteId);
  return apiUrl(`/api/evaluation/report?${q.toString()}`);
}

export async function submitHumanReview(payload) {
  const res = await fetch(apiUrl("/api/evaluation/human-review"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

/** Stream an evaluation suite, reporting live progress events. */
export async function runEvaluationStream(payload, onEvent, signal) {
  const res = await fetch(apiUrl("/api/evaluation/run/stream"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(await readError(res));

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      let event;
      try { event = JSON.parse(line.slice(5).trim()); } catch { continue; }
      if (event.type === "result") result = event.result;
      if (event.type === "error") throw new Error(event.message || "evaluation error");
      onEvent(event);
    }
  }
  if (!result) throw new Error("the evaluation ended without producing a result");
  return result;
}

/* ── Tracing & observability (Task 7) ───────────────────────── */
export const getObservabilityStatus = () => getJSON("/api/observability/status");
export const getTraces          = (limit = 20) => getJSON(`/api/observability/traces?limit=${limit}`);
export const getTraceTree       = (id) => getJSON(`/api/observability/traces/${encodeURIComponent(id)}/tree`);
export const getTraceErrors     = (limit = 10) => getJSON(`/api/observability/errors?limit=${limit}`);
export const getTraceAnalysis   = (id) => getJSON(`/api/observability/analysis/${encodeURIComponent(id)}`);
export const getTraceRootCause  = (id) => getJSON(`/api/observability/root-cause/${encodeURIComponent(id)}`);
export const getOptimizationPolicy = () => getJSON("/api/observability/policy");
export const getFailureTargets  = () => getJSON("/api/observability/failure-targets");
export const getImprovementCycles = () => getJSON("/api/observability/cycles");

async function postJSON(path, payload) {
  const res = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export const resetOptimizationPolicy  = () => postJSON("/api/observability/policy/reset");
export const revertOptimizationPolicy = () => postJSON("/api/observability/policy/revert");
export const armControlledFailure     = (payload) => postJSON("/api/observability/controlled-failure", payload);

/**
 * Stream the full self-improvement cycle: trace → diagnose → improve → re-run →
 * measure → verify. onEvent receives each stage event; resolves with the report.
 */
export async function runImprovementStream(payload, onEvent, signal) {
  const res = await fetch(apiUrl("/api/observability/improve/stream"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(await readError(res));

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let report = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      let event;
      try { event = JSON.parse(line.slice(5).trim()); } catch { continue; }
      if (event.type === "report") report = event.report;
      if (event.type === "error") throw new Error(event.message || "improvement cycle error");
      onEvent(event);
    }
  }
  if (!report) throw new Error("the improvement cycle ended without producing a report");
  return report;
}

export async function generateReport(runId, { force = false } = {}) {
  const res = await fetch(apiUrl("/api/report/generate"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id: runId, force }),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export function reportPreviewUrl(reportId, { embedded = true } = {}) {
  return apiUrl(
    `/api/report/${encodeURIComponent(reportId)}/preview?embedded=${embedded}&t=${Date.now()}`,
  );
}

/** Download via blob so API errors surface instead of navigating away. */
export async function downloadReport(reportId, format) {
  const res = await fetch(
    apiUrl(`/api/report/${encodeURIComponent(reportId)}/download/${format}`),
  );
  if (!res.ok) throw new Error(await readError(res));
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `InsightPulse-Report-${reportId}.${format}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}
