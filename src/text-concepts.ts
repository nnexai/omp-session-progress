// Public, fixed 40-concept software-engineering dictionary and L2 Ridge scorer.
// Converts prompt text (with preceding-assistant plan inheritance on brief approvals),
// remaining todos, recent tool intents, and recent assistant prose into 4 compact,
// general scalar duration priors. Zero raw vocabulary or project tokens are ever stored.

export interface TextRidgeModel {
	/** Ridge weights (intercept + 40 concepts) for prompt + approval-inherited total duration prior. */
	bP: number[];
	/** Ridge weights for remaining todo list text. */
	bT: number[];
	/** Ridge weights for recent tool intents. */
	bI: number[];
	/** Ridge weights for recent assistant prose. */
	bR: number[];
}

export const CONCEPTS: readonly RegExp[] = [
	/\b(implement|create|add|build|support|introduce|write)\b/i,
	/\b(refactor|restructure|simplify|split|extract|remove|replace|clean\s*up)\b/i,
	/\b(debug|fix|broken|fails?|failing|wrong|hangs?|regression|crash|bug|issue)\b/i,
	/\b(trace|investigate|research|analy[sz]e|diagnose|figure out|look into|root cause)\b/i,
	/\b(review|audit|inspect|evaluate|assess|compare|check)\b/i,
	/\b(test|tests|unit|spec|verify|verif\w*|validate|smoke|lint|typecheck|tsc)\b/i,
	/\b(benchmark|performance|latency|profil[ei]|experiment|accuracy|faster|optimi[sz]e|train)\b/i,
	/\b(integrat\w*|migrat\w*|port|cutover|end[- ]to[- ]end|e2e|full suite|overhaul)\b/i,
	/\b(install|deploy|release|rollout|restart|rebuild|package|publish|docker)\b/i,
	/\b(commit|push|merge|rebase|cherry[- ]pick|pull request|\bpr\b|git\s+status|git\s+diff)\b/i,
	/\b(render|visual|design|layout|dialog|ui|ux|widget|screen|css|style|theme|tui)\b/i,
	/\b(browser|playwright|emulator|hardware|device|native|desktop|android|phone|screenshot)\b/i,
	/\b(delegate|subagents?|workers?|supervis\w*|parallel|orchestrat\w*|swarm)\b/i,
	/\b(explain|summari[sz]e|describe|suggest|why|what|which|how does|where)\b/i,
	/\b(plan|proposal|strategy|approach|architecture|ideas?|roadmap|design doc)\b/i,
	/\b(document\w*|readme|changelog|guide|note|write[- ]?up|comment)\b/i,
	/\b(quick|small|only|just|single|one[- ]line|brief|simple|minor|typo|rename)\b/i,
	/\b(all|each|every|across|whole|entire|multiple|comprehensive|thorough|everything)\b/i,
	/\b(until|iterate|again|keep going|keep at|do not stop|repeat|loop|ratchet)\b/i,
	/\b(then|afterwards|after that|as well as|also|once|finally|next)\b/i,
	/\b(read[- ]only|no (?:code )?changes?|without (?:editing|changing)|do not (?:edit|change|implement))\b/i,
	/\b(go ahead|continue|proceed|yes|agreed|do it|approved|looks good|lgtm|ship it)\b/i,
	/\b(evidence|acceptance|reproduc\w*|baseline|receipt|criteria|report)\b/i,
	/\b(blocked|waiting|timeout|error|unexpected|mismatch|retry|stuck)\b/i,
	/\b(all tests pass|everything passes|verified|completed|ready to|done\b|no errors|build succeeds|cleanly)\b/i,
	/\b(still need|remaining|left to|moving on|phase\s+\d|step\s+\d|part\s+\d)\b/i,
	/\b(api|endpoint|server|backend|frontend|database|schema|migration|auth|cache|queue)\b/i,
	/\b(config|env|setup|script|cli|extension|plugin|hook|ci|workflow|pipeline)\b/i,
	/\b(file|files|directory|folder|repo|codebase|module|package|component)\b/i,
	/\b(read|reading|search|grep|find|list|inspecting|checking|exploring)\b/i,
	/\b(edit|editing|updating|modifying|patching|writing|creating|adding)\b/i,
	/\b(running|executing|building|compiling|testing|verifying|launching|waiting)\b/i,
	/\b(model|weights|feature|dataset|fold|cross[- ]val|holdout|metric|mae|score)\b/i,
	/\b(session|transcript|history|replay|checkpoint|stream|event|log)\b/i,
	/\b(clean|tidy|format|lint|unused|dead code|warning|deprecat)\b/i,
	/\b(edge case|fallback|guard|clamp|handle|robust|retry|timeout|race)\b/i,
	/\b(first|second|third|last|final|remaining|complete|finish|wrap)\b/i,
	/\b(sync|async|background|job|process|daemon|kill|wait|poll)\b/i,
	/\b(diff|patch|hunk|branch|stash|checkout|revert|reset|tag)\b/i,
	/\b(state|store|cache|index|persist|save|load|export|import)\b/i,
];

export const CONCEPT_COUNT = CONCEPTS.length;

export function conceptVec(text: string): Float64Array {
	const out = new Float64Array(CONCEPT_COUNT);
	if (!text) return out;
	const s = text.slice(0, 3000);
	for (let i = 0; i < CONCEPT_COUNT; i++) {
		if (CONCEPTS[i]!.test(s)) out[i] = 1;
	}
	return out;
}

export function conceptCounts(texts: readonly string[]): Float64Array {
	const out = new Float64Array(CONCEPT_COUNT);
	for (let k = 0; k < texts.length; k++) {
		const s = texts[k]!.slice(0, 400);
		for (let i = 0; i < CONCEPT_COUNT; i++) {
			if (CONCEPTS[i]!.test(s)) out[i]! += 1;
		}
	}
	for (let i = 0; i < CONCEPT_COUNT; i++) out[i] = Math.log1p(out[i]!);
	return out;
}

export function evalRidge(beta: readonly number[], x: Float64Array): number {
	let s = beta[0] ?? 0;
	const len = Math.min(x.length, beta.length - 1);
	for (let j = 0; j < len; j++) s += beta[j + 1]! * x[j]!;
	return s;
}

const round4 = (n: number) => Math.round(n * 10_000) / 10_000;

/** Weighted Cholesky L2 Ridge regression: min sum w_i (x_i^T b - y_i)^2 + lam * ||b[1:]||^2. */
export function fitRidge(X: Float64Array[], y: Float64Array, w: Float64Array, lam: number): number[] {
	const n = X.length;
	const d = X[0]!.length;
	const p = d + 1;
	const A = new Float64Array(p * p);
	const b = new Float64Array(p);
	for (let i = 0; i < n; i++) {
		const wi = w[i]!;
		const xi = X[i]!;
		const yi = y[i]!;
		A[0]! += wi;
		b[0]! += wi * yi;
		for (let j = 0; j < d; j++) {
			const v = xi[j]!;
			if (v === 0) continue;
			const wv = wi * v;
			A[j + 1]! += wv;
			A[(j + 1) * p]! += wv;
			b[j + 1]! += wv * yi;
			for (let k = j; k < d; k++) {
				const vk = xi[k]!;
				if (vk !== 0) A[(j + 1) * p + (k + 1)]! += wv * vk;
			}
		}
	}
	for (let j = 1; j < p; j++) {
		A[j * p + j]! += lam;
		for (let k = 0; k < j; k++) A[j * p + k] = A[k * p + j]!;
	}
	A[0]! += 1e-6;
	const L = new Float64Array(p * p);
	for (let i = 0; i < p; i++) {
		for (let j = 0; j <= i; j++) {
			let s = A[i * p + j]!;
			for (let k = 0; k < j; k++) s -= L[i * p + k]! * L[j * p + k]!;
			L[i * p + j] = i === j ? Math.sqrt(Math.max(1e-12, s)) : s / L[j * p + j]!;
		}
	}
	const z = new Float64Array(p);
	for (let i = 0; i < p; i++) {
		let s = b[i]!;
		for (let k = 0; k < i; k++) s -= L[i * p + k]! * z[k]!;
		z[i] = s / L[i * p + i]!;
	}
	const beta = new Array<number>(p);
	for (let i = p - 1; i >= 0; i--) {
		let s = z[i]!;
		for (let k = i + 1; k < p; k++) s -= L[k * p + i]! * beta[k]!;
		beta[i] = round4(s / L[i * p + i]!);
	}
	return beta;
}
