import { describe, expect, test } from 'bun:test';
import { orderedSteps, proseRoute } from './prose';

describe('proseRoute', () => {
	test('preserves prospective stages without inventing terminality', () => {
		expect(proseRoute('I’ll verify the change, then summarize.')).toEqual({
			steps: ['verify', 'report'], tail: 'unknown', conditional: false,
		});
		expect(proseRoute("I'll verify A, then implement B.")).toEqual({
			steps: ['verify', 'edit'], tail: 'more', conditional: false,
		});
		expect(proseRoute("I'll inspect the code, implement the fix, and run the tests.")).toEqual({
			steps: ['inspect', 'edit', 'verify'], tail: 'unknown', conditional: false,
		});
	});

	test('recognizes explicit terminal verification and a reporting handoff', () => {
		expect(proseRoute('All that remains is the final verification.')).toEqual({
			steps: ['verify'], tail: 'final', conditional: false,
		});
		expect(proseRoute("I'll do the final verification, then report.")).toEqual({
			steps: ['verify', 'report'], tail: 'final', conditional: false,
		});
		expect(proseRoute("I'll do the final verification, then deploy.")).toEqual({
			steps: ['verify', 'release'], tail: 'more', conditional: false,
		});
		expect(proseRoute("I'll run the final verification, then inspect the result.")).toEqual({
			steps: ['verify', 'inspect'], tail: 'unknown', conditional: false,
		});
	});

	test('reads explicitly forward-looking ordered lists', () => {
		expect(proseRoute('Next steps:\n1. Inspect the implementation\n2. Run the tests\n3. Deploy the release')).toEqual({
			steps: ['inspect', 'verify', 'release'], tail: 'more', conditional: false,
		});
	});

	test('does not promote a conditional remedy to a committed stage', () => {
		expect(proseRoute("I'll verify the change; if tests fail, I'll fix the issue.")).toEqual({
			steps: ['verify'], tail: 'unknown', conditional: true,
		});
		expect(proseRoute("I'll verify, then fix it if tests fail.")).toEqual({
			steps: ['verify'], tail: 'unknown', conditional: true,
		});
		expect(proseRoute("If tests fail, I'll fix them.")).toBeNull();
	});

	test('abstains on reports, quotations, examples, and topic bags', () => {
		for (const text of [
			'I verified the changes and implemented the fix.',
			'The user said "I will verify, then deploy".',
			'> I will verify, then deploy.',
			'For example, I will verify then deploy.',
			'```ts\nI will verify then deploy\n```',
			'Verify, implementation, tests, release.',
			"I'll inspect `verify.ts`.",
		]) {
			if (text.includes('inspect')) {
				expect(proseRoute(text)).toEqual({ steps: ['inspect'], tail: 'unknown', conditional: false });
			} else {
				expect(proseRoute(text)).toBeNull();
			}
		}
	});

	test('does not count negated work or mistake it for terminality', () => {
		expect(proseRoute("I will not verify; I'll implement the change.")).toEqual({
			steps: ['edit'], tail: 'unknown', conditional: false,
		});
		expect(proseRoute("I'll verify; this is not the final verification.")).toEqual({
			steps: ['verify'], tail: 'unknown', conditional: false,
		});
	});
});

describe('orderedSteps', () => {
	test('categorizes todo actions in order without treating filenames as actions', () => {
		expect(orderedSteps('Inspect verify.ts, implement the change, then run tests')).toEqual(['inspect', 'edit', 'verify']);
		expect(orderedSteps('Write tests, then verify the implementation')).toEqual(['edit', 'verify']);
		expect(orderedSteps('miscellaneous context')).toEqual([]);
	});
	test('limits the route to twelve categorical steps', () => {
		expect(orderedSteps(Array(20).fill('verify').join(' then '))).toEqual(Array(12).fill('verify'));
	});
});
