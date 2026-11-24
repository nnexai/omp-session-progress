// Eval host: an OMP extension, so every estimate goes through OMP's own
// `@widget` role resolution, credentials and the production transport. Loaded
// only by eval/run.ts into a disposable `omp --mode rpc` process; it registers
// /progress-eval <job.json>. Nothing is sent into any session.
//
// Per (case, rep): re-extract the historic session, and for each probe fold
// exactly the events before the cut through the production `applyEvent`,
// build the production payload, call @widget with the production system
// prompt, and read the reply with the production `readingFor`. Probes run in
// order, like the live widget's calls within one prompt.

import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { applyEvent, buildPayload, emptyFacts } from "../src/facts";
import { readingFor, SYSTEM_PROMPT } from "../src/observer";
import { type CallRecord, type Credentials, makeTransport, type Model, ROLE } from "../src/transport";
import { extractSession, type Trace } from "./extract";
import { gradeCase, type Prediction, type Probe } from "./grade";
import { EVAL_DIR } from "./paths";

export interface Case {
	id: string;
	session: string;
	prompt: number;
	tools: number;
	activeS: number;
	tags: string[];
	probes: Probe[];
}

export interface Job {
	cases: Case[];
	/** Case ids whose transcripts may be written (train split). */
	train: string[];
	reps: number;
	out: string;
	concurrency: number;
}

interface Ctx {
	ui: { notify(message: string, type?: "info" | "warning" | "error"): void };
	modelRegistry: Credentials;
	models: { resolve(spec: string): Model | undefined };
}
interface Api {
	registerCommand(name: string, options: { description: string; handler: (args: string, ctx: Ctx) => Promise<void> | void }): void;
}

const CALL_DEADLINE_MS = 30_000;
const CASE_DEADLINE_MS = 240_000;
const ATTEMPTS = 3;

class ProviderError extends Error {}

export default function progressEval(pi: Api) {
	pi.registerCommand("progress-eval", {
		description: "Grade the progress estimator on historic prompts (eval/run.ts only)",
		handler: async (args, ctx) => {
			const job: Job = JSON.parse(readFileSync(args.trim(), "utf8"));
			const model = ctx.models.resolve(ROLE);
			if (!model) {
				ctx.ui.notify(`progress-eval: failed: ${ROLE} does not resolve`, "error");
				return;
			}
			const index: Record<string, string> = JSON.parse(readFileSync(join(EVAL_DIR, "index.json"), "utf8"));
			const train = new Set(job.train);
			// Six spaced digits: unique per run, and the same token count every run, so the tag
			// itself never moves cost between variants.
			const runTag = Array.from({ length: 6 }, () => Math.floor(Math.random() * 10)).join(" ");
			mkdirSync(join(job.out, "traces"), { recursive: true });
			const resultsPath = join(job.out, "results.jsonl");
			const done = new Set(
				existsSync(resultsPath)
					? readFileSync(resultsPath, "utf8")
							.split("\n")
							.filter(Boolean)
							.map(l => {
								const row = JSON.parse(l);
								return `${row.prompt_id}#${row.rep}`;
							})
					: [],
			);
			const queue = job.cases.flatMap(c => Array.from({ length: job.reps }, (_, rep) => ({ c, rep }))).filter(({ c, rep }) => !done.has(`${c.id}#${rep}`));
			const traces = new Map<string, Trace>();
			const traceOf = (session: string) => {
				let t = traces.get(session);
				if (!t) {
					const path = index[session];
					if (!path) throw new Error(`no session file for ${session}`);
					t = extractSession(path);
					traces.set(session, t);
				}
				return t;
			};

			const runCase = async (c: Case, rep: number, deadline: AbortSignal) => {
				const trace = traceOf(c.session);
				const transcript: Array<{ role: string; content: string }> = [{ role: "system", content: SYSTEM_PROMPT }];
				const predictions: Prediction[] = [];
				const usage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
				let ms = 0;
				let served = "";
				let retries = 0;
				const notes: string[] = [];
				for (const probe of c.probes) {
					const facts = emptyFacts();
					for (const { t, event } of trace.events.slice(0, probe.cut)) applyEvent(facts, event, t);
					const payload = buildPayload(facts, probe.at);
					let record: CallRecord | undefined;
					const transport = makeTransport(ctx.modelRegistry, model, `eval-${c.id}-rep${rep}`, call => {
						record = call;
					});
					let text: string | undefined;
					for (let attempt = 1; text === undefined; attempt++) {
						if (deadline.aborted) throw new ProviderError("case deadline exceeded");
						try {
							// The trailing run tag keeps a rep from reading the cache entry an identical earlier call wrote:
							// live payloads never repeat (the observer skips unchanged ones), so neither may the eval's.
							const user = `${payload}\n(replay ${runTag} ${rep % 10})`;
							text = (await transport({ system: SYSTEM_PROMPT, user, signal: AbortSignal.any([deadline, AbortSignal.timeout(CALL_DEADLINE_MS)]) })).text;
						} catch (error) {
							if (attempt >= ATTEMPTS) throw new ProviderError(error instanceof Error ? error.message : String(error));
							retries += 1;
							await Bun.sleep(1000 * 2 ** attempt * (0.5 + Math.random()));
						}
					}
					ms += record?.ms ?? 0;
					served = record?.model ?? served;
					usage.input += record?.usage?.input ?? 0;
					usage.output += record?.usage?.output ?? 0;
					usage.cacheRead += record?.usage?.cacheRead ?? 0;
					usage.cacheWrite += record?.usage?.cacheWrite ?? 0;
					transcript.push({ role: "user", content: payload }, { role: "assistant", content: text });
					let prediction: Prediction = { progress: null, remainingMinutes: null };
					const label = `${Math.round(probe.elapsedS)}s/${Math.round(probe.elapsedS + probe.remainingS)}s`;
					try {
						prediction = readingFor(facts, text, probe.at).estimate;
					} catch (error) {
						notes.push(`probe ${label}: unparsable reply (${error instanceof Error ? error.message : String(error)})`);
					}
					predictions.push(prediction);
				}
				const grade = gradeCase(c.probes, predictions);
				const minutes = (s: number) => `${(s / 60).toFixed(1)}m`;
				const explanation = c.probes
					.map((p, i) => {
						const pred = predictions[i];
						const shown = pred?.progress == null || pred.remainingMinutes == null ? "none" : `${Math.round(pred.progress * 100)}%/${minutes(pred.remainingMinutes * 60)} left`;
						return `${minutes(p.elapsedS)} of ${minutes(p.elapsedS + p.remainingS)}: ${shown} vs ${Math.round((100 * p.elapsedS) / (p.elapsedS + p.remainingS))}%/${minutes(p.remainingS)} left`;
					})
					.concat(notes)
					.join("; ");
				const prompted = usage.input + usage.cacheRead + usage.cacheWrite;
				const row = {
					prompt_id: c.id,
					rep,
					prompt: `${c.id}: historic prompt, ${Math.round(c.activeS / 60)} active min, ${c.tools} tool calls, ${c.probes.length} probes`,
					tags: c.tags,
					grade,
					explanation,
					status: "ok",
					model: served,
					usage,
					latency_s: ms / 1000,
					cache_read_share: prompted ? usage.cacheRead / prompted : 0,
					calls: c.probes.length,
					retries,
				};
				if (train.has(c.id)) writeFileSync(join(job.out, "traces", `${c.id}_rep${rep}.json`), JSON.stringify(transcript));
				appendFileSync(resultsPath, `${JSON.stringify(row)}\n`);
			};

			const worker = async () => {
				for (let item = queue.shift(); item; item = queue.shift()) {
					const { c, rep } = item;
					const deadline = AbortSignal.timeout(CASE_DEADLINE_MS);
					try {
						await runCase(c, rep, deadline);
					} catch (error) {
						const kind = deadline.aborted ? "timeout" : error instanceof ProviderError ? "api_error" : "harness_error";
						appendFileSync(join(job.out, "errors.jsonl"), `${JSON.stringify({ prompt_id: c.id, rep, class: kind, message: (error instanceof Error ? error.message : String(error)).slice(0, 200) })}\n`);
					}
				}
			};
			await Promise.all(Array.from({ length: Math.max(1, job.concurrency) }, worker));
			ctx.ui.notify(`progress-eval: done ${model.provider}/${model.id}`, "info");
		},
	});
}
