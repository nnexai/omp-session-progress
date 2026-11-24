// The progress widget renderer: pure text + optional OMP theme styling so it
// can be tested without OMP and renders natively in both the TUI and RPC hosts.
//
// Layout (2–3 rows, borderless, aligned with OMP's native TODO/status visuals):
//   PROGRESS  ━━━━━━━━━───────────────  ~38%  ~2–4m left · 1m 24s         editing
//    ├─ todos 2/5 · 11 tools · now: bash: Running unit tests
//    └─ next: Update README and verify build

import { activeMs, type WorkPhase } from "./facts";
import type { Report } from "./observer";

export type Tone = "accent" | "muted" | "dim" | "warning" | "success";
export type Paint = (tone: Tone, text: string) => string;

export interface WidgetStyle {
	fg: Paint;
	bold?: (text: string) => string;
	filled?: string;
	empty?: string;
	branch?: string;
	last?: string;
	check?: string;
	warn?: string;
	clock?: string;
}

type Segment = { tone: Tone | null; bold?: boolean; text: string };

/** Below this width labels compact and the right-aligned phase badge drops. */
export const NARROW = 60;
/** Minimum width to show the right-aligned workflow phase badge. */
const PHASE_TAG_WIDTH = 68;

/** Format elapsed active duration concisely: "14s", "1m 28s", "12m", "1h 15m". */
export function formatElapsed(ms: number, compact = false): string {
	const totalS = Math.max(0, Math.round(ms / 1000));
	if (totalS < 60) return `${totalS}s`;
	const m = Math.floor(totalS / 60);
	const s = totalS % 60;
	if (m < 10) return compact ? `${m}m${String(s).padStart(2, "0")}s` : `${m}m ${String(s).padStart(2, "0")}s`;
	if (m < 60) return `${m}m`;
	const h = Math.floor(m / 60);
	const remM = m % 60;
	return remM ? (compact ? `${h}h${remM}m` : `${h}h ${remM}m`) : `${h}h`;
}

/** Format a single remaining duration: "<30s", "40s", "1m", "2m", "1h 10m". */
function formatUnit(minutes: number): string {
	const sec = Math.max(0, Math.round(minutes * 60));
	if (sec < 30) return "<30s";
	if (sec < 55) return `${Math.round(sec / 10) * 10}s`;
	const m = Math.max(1, Math.round(sec / 60));
	if (m < 90) return `${m}m`;
	const h = Math.floor(m / 60);
	const remM = Math.round((m % 60) / 5) * 5;
	return remM && remM < 60 ? `${h}h ${remM}m` : `${Math.round(m / 60)}h`;
}

/** Format remaining time (calibrated range when wide, single value when tight). */
export function formatRemaining(medianMin: number, lowMin: number | null, highMin: number | null, narrow: boolean): string {
	const suffix = narrow ? "" : " left";
	if (medianMin < 0.5) return `<30s${suffix}`;
	if (lowMin !== null && highMin !== null && !narrow) {
		const loM = Math.max(1, Math.round(lowMin));
		const hiM = Math.max(loM, Math.round(highMin));
		// Show range only when it communicates real spread (e.g. ~2–5m left, not ~2–2m left).
		if (medianMin >= 1 && hiM > loM && hiM <= 90 && hiM / loM >= 1.5) {
			return `~${loM}–${hiM}m${suffix}`;
		}
	}
	return `~${formatUnit(medianMin)}${suffix}`;
}

function fit(segments: Segment[], max: number): Segment[] {
	const out: Segment[] = [];
	let used = 0;
	for (const seg of segments) {
		if (!seg.text) continue;
		if (used + seg.text.length <= max) {
			out.push(seg);
			used += seg.text.length;
			continue;
		}
		if (max - used > 1) out.push({ ...seg, text: `${seg.text.slice(0, max - used - 1)}…` });
		else if (out.length) {
			const prev = out[out.length - 1]!;
			prev.text = `${prev.text.slice(0, -1)}…`;
		}
		break;
	}
	return out;
}

const visibleLen = (segments: Segment[]) => segments.reduce((n, s) => n + s.text.length, 0);

function barSegments(progress: number | null, width: number, fillTone: Tone, filledChar: string, emptyChar: string): Segment[] {
	if (width <= 0) return [];
	if (progress === null) return [{ tone: "dim", text: emptyChar.repeat(width) }];
	const clamped = Math.max(0, Math.min(1, progress));
	const filled = Math.round(clamped * width);
	const out: Segment[] = [];
	if (filled > 0) out.push({ tone: fillTone, text: filledChar.repeat(filled) });
	if (width - filled > 0) out.push({ tone: "dim", text: emptyChar.repeat(width - filled) });
	return out;
}

function phaseBadge(phase: WorkPhase, blocker: string | null, underwayTasks: number): { text: string; tone: Tone } {
	if (blocker) return { text: "blocked", tone: "warning" };
	if (phase === "waiting") return { text: "waiting", tone: "warning" };
	if (phase === "retrying") return { text: "retrying", tone: "warning" };
	if (phase === "done") return { text: "done", tone: "success" };
	if (phase === "watching") return { text: "idle", tone: "dim" };
	if (phase === "delegating" && underwayTasks > 0) {
		return { text: `delegating (${underwayTasks})`, tone: "accent" };
	}
	return { text: phase, tone: "dim" };
}

function activitySegments(report: Report, narrow: boolean): Segment[] {
	const f = report.facts;
	const parts: Segment[] = [];
	const sep = () => {
		if (parts.length) parts.push({ tone: "dim", text: " · " });
	};

	if (f.requests === 0 && !f.running) {
		parts.push({ tone: "muted", text: narrow ? "ready · local" : "ready · local estimator (no LLM calls)" });
		return parts;
	}

	// Only display todos when updated in this prompt so leftover plans never mislead.
	const liveTodos = f.todos.filter(t => t.status !== "dropped");
	if (f.todoUpdated && liveTodos.length > 0) {
		const done = liveTodos.filter(t => t.status === "done").length;
		const blocked = liveTodos.filter(t => t.status === "blocked").length;
		const todoText =
			!narrow && f.todoPhasesTotal > 1
				? `phase ${Math.min(f.todoPhasesTotal, f.todoPhasesDone + (done < liveTodos.length ? 1 : 0))}/${f.todoPhasesTotal} · todos ${done}/${liveTodos.length}`
				: narrow
					? `${done}/${liveTodos.length}`
					: `todos ${done}/${liveTodos.length}`;
		parts.push({ tone: done === liveTodos.length ? "success" : null, text: todoText });
		if (blocked > 0) {
			sep();
			parts.push({ tone: "warning", text: `${blocked} blocked` });
		}
	}

	sep();
	parts.push({ tone: "muted", text: `${f.toolCalls} tool${f.toolCalls === 1 ? "" : "s"}` });
	if (f.failedCalls > 0) {
		parts.push({ tone: "warning", text: ` (${f.failedCalls} err)` });
	}

	const running = [...f.underway.values()];
	if (f.asking.size > 0) {
		sep();
		parts.push({ tone: "warning", text: narrow ? "awaiting reply" : "waiting for your answer" });
	} else if (running.length > 0) {
		sep();
		const active = running[running.length - 1]!;
		const prefix = running.length > 1 ? `${running.length} running: ` : narrow ? "" : "now: ";
		parts.push({ tone: null, text: `${prefix}${active.label}` });
	} else if (f.recent.length > 0) {
		sep();
		const last = f.recent[f.recent.length - 1]!;
		parts.push({ tone: "muted", text: `${narrow ? "" : "last: "}${last}` });
	} else if (f.running) {
		sep();
		parts.push({ tone: "dim", text: "thinking…" });
	}

	return parts;
}

function normalizeStyle(styleOrPaint: Paint | WidgetStyle): Required<WidgetStyle> {
	if (typeof styleOrPaint === "function") {
		return {
			fg: styleOrPaint,
			bold: text => text,
			filled: "━",
			empty: "─",
			branch: "├─",
			last: "└─",
			check: "✓",
			warn: "⚠",
			clock: "",
		};
	}
	return {
		fg: styleOrPaint.fg,
		bold: styleOrPaint.bold ?? (text => text),
		filled: styleOrPaint.filled ?? "━",
		empty: styleOrPaint.empty ?? "─",
		branch: styleOrPaint.branch ?? "├─",
		last: styleOrPaint.last ?? "└─",
		check: styleOrPaint.check ?? "✓",
		warn: styleOrPaint.warn ?? "⚠",
		clock: styleOrPaint.clock ?? "",
	};
}

/** Render the widget lines for a terminal `width` columns wide; empty when off. */
export function widgetLines(report: Report, now: number, width: number, styleOrPaint: Paint | WidgetStyle): string[] {
	if (report.phase === "off") return [];
	const st = normalizeStyle(styleOrPaint);
	const narrow = width < NARROW;
	const f = report.facts;
	const r = report.reading;
	const elapsed = f.running ? activeMs(f, now) : (f.lastCompletedMs ?? f.activeMs);

	// Adaptive progress bar width
	const barWidth = width >= 92 ? 22 : width >= 72 ? 16 : width >= NARROW ? 12 : 8;
	const doneState = !f.running && f.requests > 0;
	const waitingState = f.running && f.asking.size > 0;
	const hasBlocker = Boolean(r?.blocker);

	const barTone: Tone = doneState ? "success" : waitingState || hasBlocker ? "warning" : "accent";
	const barProgress = doneState ? 1 : r?.estimate.progress ?? null;

	// ---- Row 1: PROGRESS  [bar]  ~42%  ~2–4m left · 1m 24s   [phase] ----
	const header: Segment[] = [
		{ tone: "accent", bold: true, text: "PROGRESS" },
		{ tone: null, text: narrow ? " " : "  " },
		...barSegments(barProgress, barWidth, barTone, st.filled, st.empty),
		{ tone: null, text: narrow ? " " : "  " },
	];

	if (doneState) {
		header.push({ tone: "success", bold: true, text: `${st.check} 100%` });
		header.push({ tone: "muted", text: ` · done in ${formatElapsed(elapsed, narrow)}` });
	} else if (!f.running && f.requests === 0) {
		header.push({ tone: "muted", text: "watching" });
	} else if (waitingState) {
		const pct = r?.estimate.progress !== null && r?.estimate.progress !== undefined ? `~${Math.round(r.estimate.progress * 100)}% · ` : "";
		header.push({ tone: "warning", bold: true, text: `${pct}waiting for you` });
		header.push({ tone: "muted", text: ` · ${formatElapsed(elapsed, narrow)}` });
	} else if (!r || r.estimate.progress === null || r.estimate.remainingMinutes === null) {
		const msg = r?.estimate.abstainReason === "starting" ? "starting…" : r?.estimate.abstainReason ?? "watching";
		header.push({ tone: "muted", text: msg });
		if (elapsed >= 1000) header.push({ tone: "dim", text: ` · ${formatElapsed(elapsed, narrow)}` });
	} else {
		const pct = Math.round(r.estimate.progress * 100);
		const rem = formatRemaining(r.estimate.remainingMinutes, r.estimate.lowMinutes, r.estimate.highMinutes, narrow);
		const clockPrefix = !narrow && st.clock ? `${st.clock} ` : "";
		header.push({ tone: "accent", bold: true, text: `~${pct}%` });
		header.push({ tone: null, text: narrow ? " · " : "  " });
		header.push({ tone: null, text: rem });
		header.push({ tone: "dim", text: " · " });
		header.push({ tone: "muted", text: `${clockPrefix}${formatElapsed(elapsed, narrow)}` });
	}

	// ---- Body rows ----
	const body: Segment[][] = [activitySegments(report, narrow)];
	if (r?.blocker) {
		body.push([
			{ tone: "warning", bold: true, text: `${st.warn} ` },
			{ tone: "warning", text: r.blocker },
		]);
	} else if (r?.next && f.running) {
		body.push([
			{ tone: "dim", text: narrow ? "next: " : "next: " },
			{ tone: "muted", text: r.next },
		]);
	}

	const max = Math.max(10, width - 1);
	const lines: Segment[][] = [[{ tone: null, text: " " }, ...header]];
	body.forEach((row, i) => {
		const connector = i === body.length - 1 ? `  ${st.last} ` : `  ${st.branch} `;
		lines.push([{ tone: "dim", text: connector }, ...row]);
	});

	// Right-aligned workflow phase badge on row 1 when space permits
	if (!narrow && width >= PHASE_TAG_WIDTH) {
		let underwayTasks = 0;
		for (const u of f.underway.values()) if (u.name === "task" || u.name === "wait") underwayTasks += Math.max(1, u.subtasks);
		const badge = phaseBadge(r?.phase ?? (doneState ? "done" : "watching"), r?.blocker ?? null, underwayTasks);
		const badgeText = hasBlocker ? `${st.warn} ${badge.text}` : badge.text;
		const gap = max - visibleLen(lines[0]!) - badgeText.length;
		if (gap >= 2) {
			lines[0]!.push({ tone: null, text: " ".repeat(gap) }, { tone: badge.tone, text: badgeText });
		}
	}

	return lines.map(row =>
		fit(row, max)
			.map(seg => {
				let out = seg.text;
				if (seg.bold) out = st.bold(out);
				if (seg.tone) out = st.fg(seg.tone, out);
				return out;
			})
			.join(""),
	);
}
