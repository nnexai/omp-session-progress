// Disposable RPC extension; never save model packets or replies to the OMP session.
import { completeSimple } from '@oh-my-pi/pi-ai';
import { readFileSync, writeFileSync } from 'node:fs';
import type { LunaAnswer, LunaInput } from './types';

type Model = Parameters<typeof completeSimple>[0];
interface Context {
 ui: { notify(message: string, type?: 'info' | 'warning' | 'error'): void };
 models: { resolve(spec: string): Model | undefined };
 modelRegistry: { getApiKey(model: Model, sessionId?: string, options?: { signal?: AbortSignal }): Promise<string | undefined> };
}
interface Api { registerCommand(name: string, options: { description: string; handler(args: string, ctx: Context): Promise<void> }): void }
class Failure extends Error { constructor(readonly kind: string) { super(kind); } }
const phases = new Set(['idle', 'watching', 'planning', 'exploring', 'editing', 'verifying', 'delegating', 'waiting', 'retrying', 'wrapping up', 'done']);
const categories = new Set(['investigation', 'implementation', 'verification', 'endToEnd', 'release', 'review', 'deliver']);
const recentFamilies = new Set(['execute', 'inspect', 'edit', 'delegate', 'plan', 'ask', 'other']);
const toolFamilies = new Set([...recentFamilies, 'execute-verification', 'execute-e2e']);
function number(value: unknown): value is number { return typeof value === 'number' && Number.isFinite(value) && value >= 0; }
function choices(value: unknown, allowed: Set<string>): value is string[] { return Array.isArray(value) && value.length <= 32 && value.every(x => typeof x === 'string' && allowed.has(x)); }
function packet(value: unknown): LunaInput {
 if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Failure('input_schema');
 const e = value as Record<string, unknown>;
 if (typeof e.phase !== 'string' || !phases.has(e.phase) || !choices(e.request, categories) || typeof e.hasCurrentPlan !== 'boolean' || !number(e.open) || !number(e.done) || !choices(e.openKinds, categories) || !choices(e.currentKinds, categories) || !choices(e.recent, recentFamilies) || !number(e.cleanExecAfterEdit) || !number(e.errors) || typeof e.wrapSignal !== 'boolean' || typeof e.family !== 'string' || !toolFamilies.has(e.family) || !number(e.localPostMin)) throw new Failure('input_schema');
 // Rebuild rather than forwarding the caller's object, including nested arrays.
 return { phase: e.phase as LunaInput['phase'], request: e.request as LunaInput['request'], hasCurrentPlan: e.hasCurrentPlan, open: e.open, done: e.done, openKinds: e.openKinds as LunaInput['openKinds'], currentKinds: e.currentKinds as LunaInput['currentKinds'], recent: e.recent as LunaInput['recent'], cleanExecAfterEdit: e.cleanExecAfterEdit, errors: e.errors, wrapSignal: e.wrapSignal, family: e.family, localPostMin: e.localPostMin };
}
function withAbort<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
 const { promise: result, resolve, reject } = Promise.withResolvers<T>();
 const abort = () => reject(new Failure('timeout'));
 if (signal.aborted) { abort(); return result; }
 signal.addEventListener('abort', abort, { once: true });
 promise.then(value => { signal.removeEventListener('abort', abort); resolve(value); }, error => { signal.removeEventListener('abort', abort); reject(error); });
 return result;
}
const system = 'Estimate the median active minutes of work remaining AFTER the selected currently running tool returns. Exclude all remaining runtime of that tool. The packet contains only coarse observed evidence and a local post-tool estimate, not a guarantee of completion. Respond with exactly one JSON object: {"postMin": number} for a finite number from 0 to 240 inclusive, or {"postMin": null} to abstain if evidence is insufficient. No other keys or text.';

export default function toolTiming(api: Api) {
 api.registerCommand('tool-timing', { description: 'Run private post-tool estimates', handler: async (args, ctx) => {
  let resultPath: string | undefined;
  const errors: Record<string, number> = {};
  const count = (kind: string) => { errors[kind] = (errors[kind] ?? 0) + 1; };
  try {
   const job = JSON.parse(readFileSync(args.trim(), 'utf8')) as { items: unknown[]; result: string };
   resultPath = job.result;
   if (!Array.isArray(job.items) || job.items.length > 500 || typeof resultPath !== 'string') throw new Failure('job_schema');
   const inputs = job.items.map(packet);
   const expectedModel = process.env.OMP_TOOL_TIMING_EXPECTED_MODEL;
   if (!expectedModel || expectedModel.trim() !== expectedModel) throw new Failure('expected_model_missing');
   const model = ctx.models.resolve('@widget');
   if (!model || model.id !== expectedModel) throw new Failure('requested_model_mismatch');
   const answers: Array<LunaAnswer | null> = Array(inputs.length).fill(null);
   let next = 0;
   const worker = async () => {
    for (;;) {
     const i = next++;
     if (i >= inputs.length) return;
     const start = Date.now(), signal = AbortSignal.timeout(60_000), key = `tool-timing-${crypto.randomUUID()}`;
     try {
      const apiKey = await withAbort(ctx.modelRegistry.getApiKey(model, key, { signal }), signal);
      if (!apiKey) throw new Failure('credentials');
      const reply = await withAbort(completeSimple(model, { systemPrompt: [system], messages: [{ role: 'user', content: JSON.stringify(inputs[i]), timestamp: start }] }, { apiKey, sessionId: key, maxTokens: 80, disableReasoning: true, cacheRetention: 'none', signal }), signal);
      const latencyMs = Date.now() - start;
      if (reply.model !== expectedModel) throw new Failure('served_model_mismatch');
      if (reply.stopReason === 'error' || reply.stopReason === 'aborted') throw new Failure('provider_error');
      if (reply.stopReason !== 'stop') throw new Failure('incomplete_response');
      const cost = reply.usage?.cost?.total;
      if (!number(cost)) throw new Failure('missing_cost');
      const text = reply.content.filter(block => block.type === 'text').map(block => block.text).join('');
      if (text.length > 1024) throw new Failure('invalid_response');
      let parsed: unknown;
      try { parsed = JSON.parse(text); } catch { throw new Failure('invalid_response'); }
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed) || Object.keys(parsed).length !== 1 || !('postMin' in parsed)) throw new Failure('invalid_response');
      const postMin = parsed.postMin;
      if (postMin !== null && (!number(postMin) || postMin > 240)) throw new Failure('invalid_response');
      answers[i] = { postMin, latencyMs, model: reply.model, cost };
     } catch (error) { count(error instanceof Failure ? error.kind : 'transport'); }
    }
   };
   await Promise.all(Array.from({ length: Math.min(4, inputs.length) }, () => worker()));
   writeFileSync(resultPath, JSON.stringify({ answers, errors }), { mode: 0o600 });
   ctx.ui.notify(Object.keys(errors).length ? 'tool-timing: failed' : 'tool-timing: done', Object.keys(errors).length ? 'error' : 'info');
  } catch (error) {
   count(error instanceof Failure ? error.kind : 'host');
   if (resultPath) { try { writeFileSync(resultPath, JSON.stringify({ errors }), { mode: 0o600 }); } catch { /* Parent reports a missing result. */ } }
   ctx.ui.notify('tool-timing: failed', 'error');
  }
 }});
}
