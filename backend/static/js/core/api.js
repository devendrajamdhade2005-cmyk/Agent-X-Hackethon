/* Backend client. Endpoints and payload shapes are exactly the existing ones —
   this redesign changes presentation, not the API contract. */

async function readError(res) {
  try {
    const body = await res.json();
    return body.detail || body.message || `HTTP ${res.status}`;
  } catch {
    return `HTTP ${res.status}`;
  }
}

export async function getTools() {
  const res = await fetch("/api/agent/tools");
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function getHealth() {
  const res = await fetch("/health");
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

/**
 * Stream an agent run over SSE.
 * onEvent receives every decoded event: run_started | activity | result | error | done
 * Returns the final result object.
 */
export async function runAgentStream(payload, onEvent, signal) {
  const res = await fetch("/api/agent/run/stream", {
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
  const res = await fetch("/api/agent/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function generateReport(runId, { force = false } = {}) {
  const res = await fetch("/api/report/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id: runId, force }),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export function reportPreviewUrl(reportId, { embedded = true } = {}) {
  return `/api/report/${encodeURIComponent(reportId)}/preview?embedded=${embedded}&t=${Date.now()}`;
}

/** Download via blob so API errors surface instead of navigating away. */
export async function downloadReport(reportId, format) {
  const res = await fetch(`/api/report/${encodeURIComponent(reportId)}/download/${format}`);
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
