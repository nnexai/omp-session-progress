import type { Route, Step } from './types';

// These are action words, not a general classifier: absence of a match is an
// abstention. In particular, a filename or a list of topic words is not a plan.
const actions: [Step, RegExp][] = [
	['verify', /^(?:verify|verification|validate|validation|check|checking|test|testing|run\s+(?:the\s+)?(?:tests?|checks?|suite)|retest|recheck|smoke(?:-test)?|build|lint|typecheck)\b/i],
	['edit', /^(?:implement|implementation|edit|editing|change|modify|update|write|add|remove|refactor|fix|patch|adjust|migrate|replace|create)\b/i],
	['delegate', /^(?:delegate|delegation|assign|ask\s+(?:an?\s+)?(?:agent|worker|subagent))\b/i],
	['release', /^(?:release|publish|deploy|ship|commit|push|install)\b/i],
	['inspect', /^(?:inspect|inspection|investigate|review|examine|read|explore|analy[sz]e|look\s+at|trace)\b/i],
	['report', /^(?:report|summari[sz]e|summary|handoff|hand\s+off|explain|respond|reply)\b/i],
];

// Remove material which can mention actions without being instructions. Keep
// line breaks so a block quote cannot join two otherwise separate clauses.
function clean(text: string): string {
	return text
		.replace(/```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)/g, '\n')
		.replace(/^\s*>.*$/gm, '')
		.replace(/`[^`\n]*`/g, ' ')
		.replace(/“[^”\n]*”|"[^"\n]*"|(?<!\w)'[^'\n]*'(?!\w)/g, ' ')
		.replace(/[’]/g, "'")
		.replace(/\b[\w./-]+\.[\w-]+\b/g, ' ');
}

const conditional = /\b(?:if|unless|in case|depending on|should (?:the|a|any)\b)\b/i;

// A conditional remedy is not a committed stage. Drop the condition and its
// consequence, rather than mistaking "if tests fail, fix it" for two stages.
function unconditional(text: string): string {
	return text.split(/(?<=[.!?;])\s+|\n/).map(clause => {
		const i = clause.search(conditional);
		if (i < 0) return clause;
		const prefix = clause.slice(0, i);
		// A trailing condition qualifies the preceding action ("fix if ...").
		// A comma directly before it instead starts a new conditional clause.
		if (/[,:]\s*$/.test(prefix)) return prefix;
		const separator = /(?:,|\bthen\b|\band\b)[^,]*$/i.exec(prefix);
		return separator ? prefix.slice(0, separator.index) : '';
	}).join('\n');
}

/** Categorize action phrases in order, for use with already selected todo rows. */
export function orderedSteps(text: string): Step[] {
	const source = unconditional(clean(text));
	const result: Step[] = [];
	// Scan at word boundaries, consuming recognized multi-word actions.
	for (let i = 0; i < source.length && result.length < 12; i++) {
		if (!/[a-z]/i.test(source[i]) || (i > 0 && /[a-z]/i.test(source[i - 1]))) continue;
		const rest = source.slice(i);
		const action = actions.find(([, pattern]) => pattern.test(rest));
		if (!action) continue;
		const match = action[1].exec(rest)!;
		const before = source.slice(Math.max(0, i - 45), i);
		// Do not count a negated action, a past-tense report, or a test
		// mentioned as the object of an editing action ("write test").
		if (/\b(?:not|never|no|without|don't|do not|won't|will not)\s+(?:\w+\s+){0,2}$/i.test(before)) continue;
		if (action[0] === 'verify' && /\b(?:write|add|update|edit|fix|create)\s+(?:the\s+|a\s+)?$/i.test(before)) continue;
		if (action[0] !== 'verify' && /\b(?:the|a|an)\s+$/i.test(before)) continue;
		if (action[0] === 'verify' && /^test\s+(?:results?|files?|cases?)\b/i.test(rest)) continue;
		result.push(action[0]);
		i += match[0].length - 1;
	}
	return result;
}

/** Extract only explicit, prospective assistant plans from public prose. */
export function proseRoute(text: string): Route | null {
	const source = clean(text);
	const lines = source.split('\n');
	const selected: string[] = [];
	let listPlan = false;
	for (const line of lines) {
		const trimmed = line.trim();
		if (!trimmed) { listPlan = false; continue; }
		if (/^(?:my |the )?(?:plan|next steps)\s*:\s*$/i.test(trimmed) || /\b(?:i(?:'ll| will)|we(?:'ll| will))\s+(?:do|follow|take)\s+(?:the\s+)?(?:following|these)\s*:?\s*$/i.test(trimmed)) {
			listPlan = true;
			continue;
		}
		if (listPlan && /^(?:[-*+]\s+|\d+[.)]\s+)/.test(trimmed)) {
			selected.push(trimmed);
			continue;
		}
		listPlan = false;
		// Restrict extraction to the prospective clause, not preceding reports
		// or quoted user requests on the same line.
		const future = /\b(?:i(?:'ll| will| am going to)|we(?:'ll| will| are going to))\s+|\b(?:all that remains|the only remaining (?:step|work)|the (?:final|last) step)\s+(?:is|will be)\s+/i.exec(trimmed);
		if (!future) continue;
		const preceding = trimmed.slice(0, future.index);
		if (conditional.test(preceding) || /\b(?:you (?:said|asked|wrote)|you asked me to|for example|example|such as)\b/i.test(preceding)) continue;
		selected.push(trimmed.slice(future.index));
	}
	if (!selected.length) return null;
	const plan = selected.join('\n');
	const steps = orderedSteps(plan);
	if (!steps.length) return null;
	const isConditional = conditional.test(plan);
	const verify = steps.indexOf('verify');
	let tail: Route['tail'] = 'unknown';
	if (verify >= 0) {
		if (steps.slice(verify + 1).some(step => step === 'edit' || step === 'delegate' || step === 'release')) {
			tail = 'more';
		} else if (steps.slice(verify + 1).every(step => step === 'report') &&
			!/\b(?:not|never|isn't|is not)\s+(?:the\s+)?(?:final|last)\b/i.test(plan) &&
			/\b(?:all that remains|the only remaining (?:step|work)|the (?:final|last) step|(?:final|last) (?:verification|validation|check|test))\b/i.test(plan) && !isConditional) {
			tail = 'final';
		}
	}
	return { steps, tail, conditional: isConditional };
}
