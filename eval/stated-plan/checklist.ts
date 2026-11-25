import type { TodoItem } from '../../src/facts';
import { orderedSteps } from './prose';
import type { Route, Step, Tail } from './types';

const LIMIT = 12;
const CONDITIONAL = /\b(?:if|unless|depending|possibly|perhaps|maybe|might|could|optional|when needed|as needed)\b/i;
const FINAL_VERIFY = /\b(?:final|last)\s+(?:verification|validation|check|tests?|test run|review)\b|\b(?:verify|validate|check|test)\s+(?:one\s+)?(?:final|last)\s+time\b/i;
const NEGATED_FINAL = /\b(?:not|never|isn't|is not|no)\s+(?:the\s+)?(?:final|last)\b/i;

/** The caller supplies only the current prompt's confirmed, authoritative list. */
export function checklistRoute(items: readonly TodoItem[]): Route | null {
  const active = items.findIndex(item => item.status === 'in_progress');
  const steps: Step[] = [];
  let conditional = false;
  let unresolvedBeforeActive = false;
  let hasVerify = false;
  let terminalVerify = false;
  let reportAfterVerify = false;
  let uncertainAfterVerify = false;
  let moreAfterVerify = false;
  let overflow = false;

  for (let index = 0; index < items.length; index++) {
    const item = items[index]!;
    if (item.status === 'done' || item.status === 'dropped') continue;
    const earlierPending = active >= 0 && index < active && item.status !== 'in_progress';
    const rowConditional = earlierPending || item.status === 'blocked' || CONDITIONAL.test(item.text);
    if (earlierPending) unresolvedBeforeActive = true;
    if (rowConditional) conditional = true;

    const parsed = orderedSteps(item.text);
    // An opaque row must not silently turn the preceding check into a final one.
    const row: Step[] = parsed.length ? parsed : ['other'];
    if (parsed.length >= LIMIT) overflow = true;
    const explicitFinal = FINAL_VERIFY.test(item.text) && !NEGATED_FINAL.test(item.text);
    for (const step of row) {
      if (steps.length < LIMIT) steps.push(step);
      else overflow = true;
      if (!hasVerify) {
        if (step === 'verify') {
          hasVerify = true;
          terminalVerify = explicitFinal;
        }
      } else if (step === 'edit' || step === 'delegate' || step === 'release') {
        if (rowConditional) uncertainAfterVerify = true;
        else moreAfterVerify = true;
      } else if (step === 'report') {
        reportAfterVerify = true;
      } else {
        // Another verification, inspection, or unknown action is not a terminal handoff.
        uncertainAfterVerify = true;
      }
    }
  }

  if (!steps.length) return null;
  let tail: Tail = 'unknown';
  if (hasVerify && moreAfterVerify) tail = 'more';
  else if (hasVerify && (terminalVerify || reportAfterVerify) && !uncertainAfterVerify &&
           !unresolvedBeforeActive && !conditional && !overflow) tail = 'final';
  return { steps, tail, conditional };
}
