// Local benchmark data: session selection, the dev/held-out split, and where
// traces and runs live. Everything here stays outside the repository
// (default ~/.cache/omp-session-progress-bench); only aggregate numbers and
// pseudonymous ids are ever written into the repo.

import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { extractSession, type Trace } from "./extract";

export const DATA_DIR = process.env.OMP_PROGRESS_BENCH_DIR ?? join(homedir(), ".cache", "omp-session-progress-bench");
export const SESSIONS_DIR = join(homedir(), ".omp", "agent", "sessions");

export type Split = "dev" | "heldout";
export type Stratum = "stopped" | "revised" | "long" | "short" | "todo" | "open" | "other";

export interface ManifestEntry {
	id: string;
	project: string;
	stratum: Stratum;
	split: Split;
	prompts: number;
	structured: number;
	toolCalls: number;
	hours: number;
	outcomes: Record<string, number>;
}

export interface Manifest {
	createdAt: string;
	seed: string;
	scanned: number;
	eligible: number;
	perStratum: number;
	entries: ManifestEntry[];
}

export const tracePath = (id: string) => join(DATA_DIR, "traces", `${id}.json`);
export const loadTrace = (id: string): Trace => JSON.parse(readFileSync(tracePath(id), "utf8"));
export const loadManifest = (): Manifest => JSON.parse(readFileSync(join(DATA_DIR, "manifest.json"), "utf8"));

function stratum(t: Trace): Stratum {
	const outcomes = t.prompts.map(p => p.outcome);
	const tools = t.prompts.reduce((a, p) => a + p.toolStarts.length, 0);
	if (outcomes.some(o => o === "aborted" || o === "error")) return "stopped";
	if (outcomes.includes("superseded")) return "revised";
	if (t.prompts.length >= 8 || tools >= 300) return "long";
	if (t.prompts.length <= 2 && tools <= 30) return "short";
	if (t.prompts.some(p => p.todoActive)) return "todo";
	if (!t.prompts.some(p => p.structured)) return "open";
	return "other";
}

/**
 * Scan top-level session files (subagent sessions excluded), skip live or
 * benchmark-related ones, take `perStratum` per stratum in pseudonym order,
 * and split each stratum 3:2 into dev and held-out before any tuning.
 */
export function select(perStratum: number, exclude: { modifiedWithinMs: number; cwdIncludes: string[] }): Manifest {
	const candidates: Array<{ trace: Trace; path: string }> = [];
	let scanned = 0;
	const now = Date.now();
	for (const dir of readdirSync(SESSIONS_DIR)) {
		const full = join(SESSIONS_DIR, dir);
		if (!statSync(full).isDirectory()) continue;
		for (const file of readdirSync(full)) {
			if (!file.endsWith(".jsonl")) continue;
			const path = join(full, file);
			scanned += 1;
			const st = statSync(path);
			if (now - st.mtimeMs < exclude.modifiedWithinMs || st.size > 64 * 1024 * 1024) continue;
			const head = readFileSync(path, "utf8").slice(0, 4096);
			if (exclude.cwdIncludes.some(s => head.includes(s))) continue;
			const trace = extractSession(path);
			// At least one prompt with some work, so there is something to estimate.
			if (!trace.prompts.some(p => p.toolStarts.length >= 3)) continue;
			candidates.push({ trace, path });
		}
	}
	const byStratum = new Map<Stratum, Array<{ trace: Trace; path: string }>>();
	for (const c of candidates) {
		const s = stratum(c.trace);
		byStratum.set(s, [...(byStratum.get(s) ?? []), c]);
	}
	mkdirSync(join(DATA_DIR, "traces"), { recursive: true });
	const entries: ManifestEntry[] = [];
	const privateIndex: Record<string, string> = {};
	for (const [s, list] of [...byStratum].sort(([a], [b]) => a.localeCompare(b))) {
		list.sort((a, b) => a.trace.id.localeCompare(b.trace.id));
		list.slice(0, perStratum).forEach(({ trace, path }, i) => {
			writeFileSync(tracePath(trace.id), JSON.stringify(trace));
			privateIndex[trace.id] = path;
			const outcomes: Record<string, number> = {};
			for (const p of trace.prompts) outcomes[p.outcome] = (outcomes[p.outcome] ?? 0) + 1;
			const times = trace.events.map(e => e.t);
			entries.push({
				id: trace.id,
				project: trace.project,
				stratum: s,
				split: i % 5 < 3 ? "dev" : "heldout",
				prompts: trace.prompts.length,
				structured: trace.prompts.filter(p => p.structured).length,
				toolCalls: trace.prompts.reduce((a, p) => a + p.toolStarts.length, 0),
				hours: times.length ? (Math.max(...times) - Math.min(...times)) / 3_600_000 : 0,
				outcomes,
			});
		});
	}
	const manifest: Manifest = { createdAt: new Date().toISOString(), seed: "sha256 pseudonym order", scanned, eligible: candidates.length, perStratum, entries };
	writeFileSync(join(DATA_DIR, "manifest.json"), JSON.stringify(manifest, null, 2));
	// Pseudonym → session file, for local re-extraction only; never leaves this directory.
	writeFileSync(join(DATA_DIR, "private-index.json"), JSON.stringify(privateIndex, null, 2));
	return manifest;
}

export const ensureDir = (dir: string) => {
	if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
	return dir;
};
