// Benchmark host: an OMP extension so the replay calls @widget through OMP's
// own role resolution, credentials and the production transport. Loaded only
// by bench/cli.ts into a disposable `omp --mode rpc` process; it registers
// /progress-bench <job.json>, replays the job's traces and writes one result
// file per trace. Nothing is sent into any session.

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { type CallRecord, type Credentials, makeTransport, type Model, ROLE } from "../src/transport";
import { loadTrace } from "./data";
import { type Caller, replay } from "./replay";
import { variant } from "./variants";

export interface Job {
	variant: string;
	traces: string[];
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

export default function progressBench(pi: Api) {
	pi.registerCommand("progress-bench", {
		description: "Replay benchmark traces through the progress estimator (bench/cli.ts only)",
		handler: async (args, ctx) => {
			const job: Job = JSON.parse(readFileSync(args.trim(), "utf8"));
			const v = variant(job.variant);
			const model = ctx.models.resolve(ROLE);
			if (!model) {
				ctx.ui.notify(`progress-bench: failed: ${ROLE} does not resolve`, "error");
				return;
			}
			const queue = [...job.traces];
			const errors: string[] = [];
			const worker = async () => {
				for (let id = queue.shift(); id; id = queue.shift()) {
					const trace = loadTrace(id);
					const caller: Caller = async request => {
						let record: CallRecord | undefined;
						const transport = makeTransport(ctx.modelRegistry, model, `bench-${v.name}-${id}`, call => {
							record = call;
						});
						const result = await transport({ ...request, signal: AbortSignal.any([request.signal, AbortSignal.timeout(CALL_DEADLINE_MS)]) });
						return { ...result, usage: record?.usage, ms: record?.ms, model: record?.model };
					};
					try {
						const result = await replay(id, trace.events, v, caller);
						writeFileSync(join(job.out, `${id}.json`), JSON.stringify({ ...result, model: `${model.provider}/${model.id}` }));
					} catch (error) {
						errors.push(`${id}: ${error instanceof Error ? error.message : String(error)}`);
					}
				}
			};
			await Promise.all(Array.from({ length: Math.max(1, job.concurrency) }, worker));
			writeFileSync(join(job.out, "done.json"), JSON.stringify({ model: `${model.provider}/${model.id}`, errors }));
			ctx.ui.notify(`progress-bench: done ${job.traces.length - errors.length}/${job.traces.length}`, errors.length ? "warning" : "info");
		},
	});
}
