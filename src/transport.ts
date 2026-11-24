// The @widget transport: one bounded, reasoning-free completion per observer
// call, through OMP's own role resolution and credentials. Shared by the
// extension and the offline replay benchmark, so both measure the same call.

import { completeSimple } from "@oh-my-pi/pi-ai";
import type { Transport } from "./observer";

export const ROLE = "@widget";
export const MAX_OUTPUT_TOKENS = 240;

export interface Model {
	provider: string;
	id: string;
}

export interface Credentials {
	getApiKey(model: Model, sessionId?: string, options?: { signal?: AbortSignal }): Promise<string | undefined>;
}

/** Provider usage as reported by pi-ai; Anthropic counts cached prompt tokens apart from `input`. */
export interface Usage {
	input?: number;
	output?: number;
	cacheRead?: number;
	cacheWrite?: number;
	totalTokens?: number;
	cost?: { total?: number };
}

/** One finished call, for logs and measurements. */
export interface CallRecord {
	startedAt: number;
	ms: number;
	model: string;
	sent: string;
	stopReason?: string;
	error?: string;
	reply: string;
	usage?: Usage;
}

export function makeTransport(credentials: Credentials, model: Model, sessionId: string, onCall?: (call: CallRecord) => void): Transport {
	// A key of its own, so the observer never shares a provider cache lineage with the session.
	const key = `${sessionId}:session-progress`;
	return async ({ system, user, signal }) => {
		const started = Date.now();
		const apiKey = await credentials.getApiKey(model, key, { signal });
		if (!apiKey) throw new Error(`no credentials for ${model.provider}`);
		const reply = await completeSimple(
			model,
			{ systemPrompt: [system], messages: [{ role: "user", content: user, timestamp: started }] },
			{ apiKey, sessionId: key, maxTokens: MAX_OUTPUT_TOKENS, disableReasoning: true, cacheRetention: "none", signal },
		);
		const text = reply.content
			.filter((block: { type: string }) => block.type === "text")
			.map((block: { text: string }) => block.text)
			.join("");
		const usage: Usage | undefined = reply.usage;
		onCall?.({
			startedAt: started,
			ms: Date.now() - started,
			// The model the provider reports serving, else the one requested.
			model: `${model.provider}/${reply.model || model.id}`,
			sent: user,
			stopReason: reply.stopReason,
			error: reply.errorMessage,
			reply: text,
			usage,
		});
		if (reply.stopReason === "error" || reply.stopReason === "aborted") throw new Error(reply.errorMessage || reply.stopReason);
		return { text, inputTokens: (usage?.input ?? 0) + (usage?.cacheRead ?? 0) + (usage?.cacheWrite ?? 0), outputTokens: usage?.output };
	};
}
