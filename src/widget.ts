// The progress widget's content: pure text so it can be tested without OMP.
// index.ts draws these lines in a native TUI widget above the editor
// (ctx.ui.setWidget); RPC hosts get the same lines unstyled.
//
// Layout (2-4 rows, borderless, like OMP's TODO block):
//    PROGRESS  prompt · est: middle · 8s ago                    @widget
//     ├─ observed: todos 2/5 · 6 tools · now: bash: Running sum script
//     ├─ estimate: 2 of 5 requested steps done; sum.sh running
//     └─ next? (pred): Note the sum, then run the missing-file check
// "observed" is counted from events; "estimate" and "next?" come from the
// @widget model and are labeled as such. No pips or bars: stages are words.

import type { Report } from "./observer";

export type Tone = "accent" | "muted" | "dim" | "warning" | "success";
export type Paint = (tone: Tone, text: string) => string;
type Segment = [Tone | null, string];

/** Below this width labels shorten and the scope/role tags go. */
export const NARROW = 60;
/** From this width the role tag is right-aligned on the header. */
const ROLE_TAG_WIDTH = 76;
/** A reading older than this while the agent works is marked stale. */
export const STALE_MS = 30_000;

function age(ms: number): string {
	const s = Math.max(0, Math.round(ms / 1000));
	return s < 5 ? "just now" : s < 90 ? `${s}s ago` : `${Math.round(s / 60)}m ago`;
}

/** Cut segments to `max` visible characters, ending in "…" when cut. */
function fit(segments: Segment[], max: number): Segment[] {
	const out: Segment[] = [];
	let used = 0;
	for (const [tone, text] of segments) {
		if (used + text.length <= max) {
			out.push([tone, text]);
			used += text.length;
			continue;
		}
		if (max - used > 1) out.push([tone, `${text.slice(0, max - used - 1)}…`]);
		else if (out.length) out[out.length - 1]![1] = `${out[out.length - 1]![1].slice(0, -1)}…`;
		break;
	}
	return out;
}

const length = (segments: Segment[]) => segments.reduce((n, [, text]) => n + text.length, 0);

function observed(report: Report): string {
	const f = report.facts;
	const parts: string[] = [];
	if (!f.running) parts.push("idle");
	if (f.retry) parts.push(f.retry);
	const live = f.todos.filter(t => t.status !== "dropped");
	if (live.length) parts.push(`todos ${live.filter(t => t.status === "done").length}/${live.length}`);
	parts.push(`${f.toolCalls} tool${f.toolCalls === 1 ? "" : "s"}${f.failedCalls ? ` (${f.failedCalls} err)` : ""}`);
	const running = [...f.underway.values()];
	if (running.length) parts.push(`now: ${running[running.length - 1]}`);
	else if (f.recent.length) parts.push(`last: ${f.recent[f.recent.length - 1]}`);
	return parts.join(" · ");
}

/** The widget's lines for a terminal `width` columns wide; empty when off. */
export function widgetLines(report: Report, now: number, width: number, paint: Paint): string[] {
	if (report.phase === "off") return [];
	const narrow = width < NARROW;
	const f = report.facts;
	const r = report.reading;
	const stale = !!r && (report.phase !== "on" || (f.running && now - r.at > STALE_MS));
	const predicted: Tone = stale ? "dim" : "muted";

	// Header: scope, stage or abstention, freshness.
	const header: Segment[] = [["accent", "PROGRESS"]];
	const scope = r?.estimate.scope === "session" ? "session" : narrow ? null : "prompt";
	header.push([null, scope ? `  ${scope} · ` : " · "]);
	if (!r) header.push(["muted", report.inFlight ? "estimating…" : "watching"]);
	else {
		if (!f.running) header.push([null, "turn ended · "]);
		header.push(r.estimate.stage ? ["accent", `est: ${r.estimate.stage}`] : ["muted", "no estimate"]);
		if (r.blocker) header.push(["warning", " · ⚠ blocked"]);
		header.push([stale ? "warning" : "muted", ` · ${stale && report.phase === "on" ? `stale ${age(now - r.at).replace(" ago", "")}` : age(now - r.at)}`]);
		if (report.inFlight) header.push(["muted", " · updating…"]);
	}
	if (report.phase === "paused") header.push(["warning", ` · paused after ${report.failures} errors, /progress on retries`]);
	if (report.phase === "capped") header.push(["warning", " · call cap reached"]);

	// Body: observed facts, then the model's estimate or blocker, then its next-step guess.
	const label = (wide: string, short: string) => (narrow ? short : wide);
	const body: Segment[][] = [[["dim", label("observed: ", "obs: ")], [null, observed(report)]]];
	if (r) {
		const estimate: Segment[] = r.estimate.stage
			? [["dim", label("estimate: ", "est: ")], [predicted, r.estimate.basis]]
			: [["dim", label("estimate: ", "est: ")], [predicted, `not enough evidence to estimate — ${r.estimate.basis}`]];
		const next: Segment[] | null =
			r.next === "unknown" ? null : [["dim", label("next? (pred): ", "next?: ")], [predicted, r.next]];
		if (r.blocker) {
			body.push([["warning", label("blocker (est): ", "blocker?: ")], [stale ? "dim" : "warning", r.blocker]]);
			body.push(next ?? estimate);
		} else {
			body.push(estimate);
			if (next) body.push(next);
		}
	}

	const max = Math.max(10, width - 1);
	const lines: Segment[][] = [[[null, " "], ...header]];
	body.forEach((row, i) => lines.push([[null, i === body.length - 1 ? "  └─ " : "  ├─ "], ...row]));
	if (!narrow && width >= ROLE_TAG_WIDTH) {
		const gap = max - length(lines[0]!) - "@widget".length;
		if (gap >= 2) lines[0]!.push([null, " ".repeat(gap)], ["dim", "@widget"]);
	}
	return lines.map(row => fit(row, max).map(([tone, text]) => (tone ? paint(tone, text) : text)).join(""));
}
