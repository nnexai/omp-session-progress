// Shared deterministic feature extraction from `Facts` at timestamp `now`.
// Used identically by both the live local estimator and offline training/eval
// to guarantee zero train/serve skew.

import { activeMs, EXPLORE_TOOLS, EXEC_TOOLS, type Facts, WRITE_TOOLS } from "./facts";

export interface ExtractedFeatures {
	/** 78 numeric features fed into the local GBDT ensemble. */
	x: number[];
	/** Active elapsed seconds of the current prompt. */
	elapsedS: number;
	/** ln(elapsedS + 30): baseline scale-equivariant anchor. */
	logElapsed: number;
	/** Todo-velocity anchor when active todos exist, else logElapsed. */
	anchorZ: number;
	/** True when the agent actively maintains a todo list in this prompt. */
	hasActiveTodo: boolean;
	/** Count of todos completed during this prompt. */
	todoDoneThisPrompt: number;
}

export const SMOOTH_S = 30;

export function extractFeatures(facts: Facts, now: number): ExtractedFeatures {
	const elapsedS = activeMs(facts, now) / 1000;
	const logElapsed = Math.log(elapsedS + SMOOTH_S);

	const liveTodos = facts.todos.filter(t => t.status !== "dropped");
	const todoTotal = liveTodos.length;
	const todoDone = liveTodos.filter(t => t.status === "done").length;
	const todoInProg = liveTodos.filter(t => t.status === "in_progress").length;
	const todoBlocked = liveTodos.filter(t => t.status === "blocked").length;
	const remTodoItems = liveTodos.filter(t => t.status !== "done");
	const todoRem = remTodoItems.length;
	const hasActiveTodo = facts.todoUpdated && todoTotal > 0;
	const hasTodoNum = hasActiveTodo ? 1 : 0;
	const todoFrac = hasActiveTodo ? todoDone / todoTotal : 0;
	const todoDoneThisPrompt = facts.todoDoneEventsS.length;
	const denomPrompt = Math.max(1, facts.todoStartRem, todoDoneThisPrompt + todoRem);
	const promptTodoFrac = hasActiveTodo ? todoDoneThisPrompt / denomPrompt : 0;
	const todoAllDone = hasActiveTodo && todoRem === 0 ? 1 : 0;
	const todoBaseS = facts.todoInitAtS ?? 0;
	const lastTodoDoneS = todoDoneThisPrompt > 0 ? facts.todoDoneEventsS[todoDoneThisPrompt - 1]! : todoBaseS;
	const timeSinceTodoDoneS = hasActiveTodo ? Math.max(0, elapsedS - lastTodoDoneS) : 0;
	const timeSinceTodoStartS = hasActiveTodo ? Math.max(0, elapsedS - todoBaseS) : elapsedS;

	// Intra-todo progress from tools executed since the last todo completion
	const lastTodoTools = todoDoneThisPrompt > 0 ? facts.todoDoneToolsAt[todoDoneThisPrompt - 1]! : 0;
	const toolsSinceTodoDone = hasActiveTodo ? Math.max(0, facts.toolCalls - lastTodoTools) : 0;
	const avgToolsPerTodo = todoDoneThisPrompt > 0 ? Math.max(2, lastTodoTools / todoDoneThisPrompt) : 8;
	const curTodoToolCredit = hasActiveTodo && todoRem > 0 ? Math.min(0.85, toolsSinceTodoDone / avgToolsPerTodo) : 0;
	const promptTodoSmoothFrac = hasActiveTodo
		? Math.min(0.99, (todoDoneThisPrompt + Math.max(0.25 * todoInProg, curTodoToolCredit)) / denomPrompt)
		: 0;

	let ewmaTodoDur = 90;
	if (todoDoneThisPrompt > 0) {
		let prev = todoBaseS;
		let acc = 0;
		for (let i = 0; i < todoDoneThisPrompt; i++) {
			const dt = Math.max(5, facts.todoDoneEventsS[i]! - prev);
			acc = i === 0 ? dt : 0.45 * dt + 0.55 * acc;
			prev = facts.todoDoneEventsS[i]!;
		}
		ewmaTodoDur = (todoDoneThisPrompt * acc + 1.5 * 90) / (todoDoneThisPrompt + 1.5);
	}

	let todoImpliedLogRem = logElapsed;
	let todoEwmaImpliedLogRem = logElapsed;
	let todoVelocityLog = Math.log(90 + 10);
	if (hasActiveTodo) {
		if (todoRem === 0) {
			todoImpliedLogRem = Math.log(12 + SMOOTH_S);
			todoEwmaImpliedLogRem = Math.log(12 + SMOOTH_S);
			todoVelocityLog = Math.log(10 + 10);
		} else if (todoDoneThisPrompt > 0) {
			const avgPerTodo = (timeSinceTodoStartS + 1.5 * 90) / (todoDoneThisPrompt + 1.5);
			todoVelocityLog = Math.log(avgPerTodo + 10);
			const credit = Math.max(Math.min(timeSinceTodoDoneS, 0.75 * avgPerTodo), curTodoToolCredit * 0.65 * avgPerTodo);
			todoImpliedLogRem = Math.log(Math.max(10, todoRem * avgPerTodo - credit) + SMOOTH_S);
			const creditE = Math.max(Math.min(timeSinceTodoDoneS, 0.75 * ewmaTodoDur), curTodoToolCredit * 0.65 * ewmaTodoDur);
			todoEwmaImpliedLogRem = Math.log(Math.max(10, todoRem * ewmaTodoDur - creditE) + SMOOTH_S);
		} else {
			const avgPerTodo = Math.max(45, elapsedS * 0.9);
			todoVelocityLog = Math.log(avgPerTodo + 10);
			const estRem = Math.max(20, todoRem * avgPerTodo - Math.min(timeSinceTodoDoneS, 0.5 * avgPerTodo));
			todoImpliedLogRem = Math.log(estRem + SMOOTH_S);
			todoEwmaImpliedLogRem = todoImpliedLogRem;
		}
	}
	const todoRatioImplied =
		hasActiveTodo && promptTodoSmoothFrac > 0.05
			? Math.log(Math.max(10, (timeSinceTodoStartS * (1 - promptTodoSmoothFrac)) / promptTodoSmoothFrac) + SMOOTH_S)
			: logElapsed;
	const anchorZ = hasActiveTodo ? todoImpliedLogRem : logElapsed;

	const remTodoChars = hasActiveTodo ? remTodoItems.reduce((a, t) => a + t.text.length, 0) : 0;
	const remHeavyTodos = hasActiveTodo
		? remTodoItems.filter(t => /\b(implement|build|refactor|migrate|create|add|integrate|rewrite|audit|investigate|e2e|browser|benchmark)\b/i.test(t.text)).length
		: 0;
	const todoPhasesRem = hasActiveTodo ? Math.max(0, facts.todoPhasesTotal - facts.todoPhasesDone) : 0;
	const phaseFrac = hasActiveTodo && facts.todoPhasesTotal > 0 ? facts.todoPhasesDone / facts.todoPhasesTotal : 0;

	const tools = facts.toolCalls;
	const logTools = Math.log1p(tools);
	const logSecPerTool = Math.log(elapsedS / Math.max(1, tools) + 5);
	const avgToolDurLog = Math.log(facts.toolTimeSumS / Math.max(1, tools) + 1);
	const maxToolDurLog = Math.log(facts.maxToolDurS + 1);
	const toolTimeShare = Math.min(1, facts.toolTimeSumS / Math.max(1, elapsedS));
	const failRate = tools > 0 ? facts.failedCalls / tools : 0;
	const exploreShare = tools > 0 ? facts.catCounts.explore / tools : 0;
	const writeShare = tools > 0 ? facts.catCounts.write / tools : 0;
	const execShare = tools > 0 ? facts.catCounts.exec / tools : 0;
	const delegateShare = tools > 0 ? facts.catCounts.delegate / tools : 0;

	const rec = facts.toolHistory.slice(-5);
	const recN = Math.max(1, rec.length);
	const recExplore = rec.filter(x => EXPLORE_TOOLS.has(x.name)).length / recN;
	const recWrite = rec.filter(x => WRITE_TOOLS.has(x.name)).length / recN;
	const recExec = rec.filter(x => EXEC_TOOLS.has(x.name)).length / recN;
	const recErr = rec.filter(x => x.isError).length / recN;
	const recAvgDurLog = Math.log(rec.reduce((a, x) => a + x.durS, 0) / recN + 1);
	const toolsLast60s = facts.toolHistory.filter(x => elapsedS - x.atS <= 60).length;
	const turnsLast60s = facts.turnTimesS.filter(t => elapsedS - t <= 60).length;
	const timeSinceLastToolEndLog = Math.log(Math.max(0, elapsedS - facts.lastToolEndAtS) + 5);

	let underwayTaskCount = 0;
	let maxUnderwayDurS = 0;
	for (const u of facts.underway.values()) {
		if (u.name === "task" || u.name === "wait") underwayTaskCount += Math.max(1, u.subtasks);
		maxUnderwayDurS = Math.max(maxUnderwayDurS, Math.max(0, elapsedS - u.startedAtS));
	}
	const avgTaskDurLog = facts.taskDoneCount > 0 ? Math.log(facts.taskTimeSumS / facts.taskDoneCount + 5) : 0;

	const hasWritten = facts.firstWriteAtS !== null ? 1 : 0;
	const writeRatioElapsed = facts.firstWriteAtS !== null && elapsedS > 1 ? facts.firstWriteAtS / elapsedS : 1;
	const timeSinceLastWriteS = facts.lastWriteAtS !== null ? Math.max(0, elapsedS - facts.lastWriteAtS) : 0;
	const postWriteClean = facts.postWriteCleanExecCount > 0 ? 1 : 0;

	const turns = facts.assistantTurns;
	const logTurns = Math.log1p(turns);
	const logSecPerTurn = Math.log(elapsedS / Math.max(1, turns) + 5);
	const toolsPerTurn = tools / Math.max(1, turns);
	const logOutTokens = Math.log1p(facts.totalOutputTokens);
	const logOutPerTurn = Math.log1p(facts.totalOutputTokens / Math.max(1, turns));
	const logCtxTokens = Math.log1p(facts.latestContextTokens);
	const isFastModel = /luna|haiku|flash|mini/i.test(facts.model) ? 1 : 0;
	const isSlowModel = /sol|opus|pro/i.test(facts.model) ? 1 : 0;

	const logPromptChars = Math.log1p(facts.promptChars);
	const promptIdx = Math.max(0, facts.promptIndex);
	const isShortFollowup = promptIdx > 0 && facts.promptChars < 120 ? 1 : 0;
	const isFirstPrompt = promptIdx === 0 ? 1 : 0;

	const priorN = facts.priorPromptDurationsS.length;
	const priorMeanLogDur =
		priorN > 0 ? facts.priorPromptDurationsS.reduce((a, d) => a + Math.log(d + SMOOTH_S), 0) / priorN : Math.log(320 + SMOOTH_S);
	const lastPriorLogDur =
		priorN > 0 ? Math.log(facts.priorPromptDurationsS[priorN - 1]! + SMOOTH_S) : Math.log(320 + SMOOTH_S);
	const logSessionActive = Math.log(facts.sessionActiveS + SMOOTH_S);

	const x = [
		logElapsed,
		hasTodoNum,
		facts.todoInitThisPrompt ? 1 : 0,
		todoFrac,
		promptTodoFrac,
		promptTodoSmoothFrac,
		todoAllDone,
		Math.log1p(todoRem),
		Math.log1p(todoTotal),
		todoInProg,
		todoBlocked,
		todoImpliedLogRem,
		todoEwmaImpliedLogRem,
		todoRatioImplied,
		todoVelocityLog,
		Math.log(timeSinceTodoDoneS + SMOOTH_S),
		curTodoToolCredit,
		Math.log1p(toolsSinceTodoDone),
		todoPhasesRem,
		phaseFrac,
		Math.log1p(remTodoChars),
		remHeavyTodos,
		facts.todoAppendCount,
		logTools,
		logSecPerTool,
		avgToolDurLog,
		maxToolDurLog,
		toolTimeShare,
		failRate,
		facts.errorStreak,
		exploreShare,
		writeShare,
		execShare,
		delegateShare,
		recExplore,
		recWrite,
		recExec,
		recErr,
		recAvgDurLog,
		toolsLast60s,
		turnsLast60s,
		timeSinceLastToolEndLog,
		facts.underway.size,
		underwayTaskCount,
		Math.log(maxUnderwayDurS + 10),
		facts.taskSubagentsLaunched,
		Math.max(0, facts.taskSubagentsLaunched - facts.taskCompleted),
		avgTaskDurLog,
		hasWritten,
		writeRatioElapsed,
		Math.log(timeSinceLastWriteS + SMOOTH_S),
		facts.postWriteExecCount,
		postWriteClean,
		facts.lastExecClean ? 1 : 0,
		facts.askCount,
		logTurns,
		logSecPerTurn,
		toolsPerTurn,
		logOutTokens,
		logOutPerTurn,
		logCtxTokens,
		isFastModel,
		isSlowModel,
		logPromptChars,
		facts.promptListItems,
		Math.min(20, promptIdx),
		isFirstPrompt,
		isShortFollowup,
		facts.promptKwHeavy ? 1 : 0,
		facts.promptKwLight ? 1 : 0,
		facts.promptKwContinue ? 1 : 0,
		facts.agentWrapUp ? 1 : 0,
		Math.min(10, priorN),
		priorMeanLogDur,
		lastPriorLogDur,
		logSessionActive,
		todoImpliedLogRem - logElapsed,
		todoRatioImplied - logElapsed,
	];

	return { x, elapsedS, logElapsed, anchorZ, hasActiveTodo, todoDoneThisPrompt };
}
