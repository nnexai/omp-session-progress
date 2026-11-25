// Shared deterministic feature extraction from `Facts` at timestamp `now`.
// Used identically by both the live local estimator and offline training/eval
// to guarantee zero train/serve skew.

import { activeMs, EXPLORE_TOOLS, EXEC_TOOLS, type Facts, WRITE_TOOLS } from "./facts";
import { conceptCounts, conceptVec, evalRidge, type TextRidgeModel } from "./text-concepts";

export interface ExtractedFeatures {
	/** 150 general, tool-agnostic causal numeric features fed into the local GBDT + SurvivalNet ensemble. */
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
	/** Tool calls completed so far in this prompt. */
	toolCalls: number;
}

export const SMOOTH_S = 30;

const RE_TODO_VERIFY = /\b(test|verify|check|commit|doc|readme|summary|reply)\b/i;

export function extractFeatures(facts: Facts, now: number, textModel?: TextRidgeModel): ExtractedFeatures {
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
	let rawAvgPerTodo = 90;
	let medianTodoDur = 90;
	if (hasActiveTodo && todoDoneThisPrompt > 0) {
		rawAvgPerTodo = Math.max(5, timeSinceTodoStartS / todoDoneThisPrompt);
		let prev = todoBaseS;
		let acc = 0;
		const dts: number[] = [];
		for (let i = 0; i < todoDoneThisPrompt; i++) {
			const dt = Math.max(5, facts.todoDoneEventsS[i]! - prev);
			acc = i === 0 ? dt : 0.45 * dt + 0.55 * acc;
			dts.push(Math.max(3, facts.todoDoneEventsS[i]! - prev));
			prev = facts.todoDoneEventsS[i]!;
		}
		ewmaTodoDur = (todoDoneThisPrompt * acc + 1.5 * 90) / (todoDoneThisPrompt + 1.5);
		dts.sort((a, b) => a - b);
		medianTodoDur = dts[dts.length >> 1]!;
	} else if (hasActiveTodo) {
		rawAvgPerTodo = Math.max(30, elapsedS);
		medianTodoDur = rawAvgPerTodo;
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

	const rawTodoLogRem = hasActiveTodo
		? todoRem === 0
			? Math.log(12 + SMOOTH_S)
			: Math.log(Math.max(10, todoRem * rawAvgPerTodo) + SMOOTH_S)
		: logElapsed;
	const medTodoLogRem = hasActiveTodo
		? todoRem === 0
			? Math.log(12 + SMOOTH_S)
			: Math.log(Math.max(10, todoRem * medianTodoDur) + SMOOTH_S)
		: logElapsed;
	const curTodoExcess = hasActiveTodo && todoRem > 0 ? Math.min(8, timeSinceTodoDoneS / Math.max(15, rawAvgPerTodo)) : 0;
	const todoScopeGrowth = facts.todoInitTotal > 0 ? todoTotal / facts.todoInitTotal : 1;
	const remVerifyTodos = hasActiveTodo ? remTodoItems.filter(t => RE_TODO_VERIFY.test(t.text)).length : 0;
	const remVerifyShare = hasActiveTodo && todoRem > 0 ? remVerifyTodos / todoRem : 0;

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
	let underwayExecCount = 0;
	let underwayExecAgeS = 0;
	let underwayWaitOrTask = 0;
	for (const u of facts.underway.values()) {
		const age = Math.max(0, elapsedS - u.startedAtS);
		if (u.name === "task" || u.name === "wait") {
			underwayTaskCount += Math.max(1, u.subtasks);
			underwayWaitOrTask += 1;
		}
		if (EXEC_TOOLS.has(u.name)) {
			underwayExecCount += 1;
			underwayExecAgeS = Math.max(underwayExecAgeS, age);
		}
		maxUnderwayDurS = Math.max(maxUnderwayDurS, age);
	}
	const avgTaskDurLog = facts.taskDoneCount > 0 ? Math.log(facts.taskTimeSumS / facts.taskDoneCount + 5) : 0;

	let oldestBgAgeS = 0;
	let bgTaskRunning = 0;
	let bgBashRunning = 0;
	let bgSlowRunning = 0;
	for (const job of facts.bgJobs.values()) {
		oldestBgAgeS = Math.max(oldestBgAgeS, Math.max(0, elapsedS - job.startedAtS));
		if (job.kind === "task") bgTaskRunning += 1;
		else if (job.kind === "bash") bgBashRunning += 1;
		if (job.isSlowModel) bgSlowRunning += 1;
	}
	const avgBgDurS = facts.bgDoneCount > 0 ? facts.bgDoneDurSumS / facts.bgDoneCount : 0;

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

	// Working set / file dynamics
	const nRead = facts.filesRead.size;
	const nEdit = facts.filesEdited.size;
	let totalFileEdits = 0;
	for (const c of facts.editPathCounts.values()) totalFileEdits += c;
	const reEditRatio = nEdit > 0 ? totalFileEdits / nEdit : 0;
	const readNovelty =
		facts.recentReadNovel.length > 0
			? facts.recentReadNovel.reduce((a, b) => a + b, 0) / facts.recentReadNovel.length
			: 1;

	// Context growth, thinking latency, tool fanout & velocity trend
	const deltaCtx = Math.max(0, facts.latestContextTokens - facts.promptStartContextTokens);
	const avgThinkS = facts.thinkCount > 0 ? facts.thinkTimeSumS / facts.thinkCount : 0;
	const recThinkS =
		facts.recentThinkS.length > 0
			? facts.recentThinkS.reduce((a, b) => a + b, 0) / facts.recentThinkS.length
			: 0;
	const avgFanout =
		facts.turnFanoutHist.length > 0
			? facts.turnFanoutHist.reduce((a, b) => a + b, 0) / facts.turnFanoutHist.length
			: 0;
	const recOutAvg =
		facts.recentOutputTokens.length > 0
			? facts.recentOutputTokens.reduce((a, b) => a + b, 0) / facts.recentOutputTokens.length
			: 0;
	const halfS = elapsedS * 0.5;
	let toolsSecondHalf = 0;
	for (let i = 0; i < facts.toolTimesS.length; i++) {
		if (facts.toolTimesS[i]! >= halfS) toolsSecondHalf++;
	}
	const toolHalfRatio = facts.toolTimesS.length > 0 ? toolsSecondHalf / facts.toolTimesS.length : 0.5;

	// 4-channel public-concept Ridge duration priors (strictly local, zero private tokens stored)
	let zp = logElapsed;
	let zt = logElapsed;
	let zi = logElapsed;
	let zr = logElapsed;
	if (textModel) {
		const effPrompt =
			facts.isApprovalPrompt && facts.prevAssistText ? `${facts.promptText}\n${facts.prevAssistText}` : facts.promptText;
		const remTodoTexts = hasActiveTodo ? remTodoItems.map(t => t.text) : [];
		zp = evalRidge(textModel.bP, conceptVec(effPrompt));
		zt = evalRidge(textModel.bT, conceptCounts(remTodoTexts));
		zi = evalRidge(textModel.bI, conceptCounts(facts.recentIntents));
		zr = evalRidge(textModel.bR, conceptCounts(facts.recentProse));
	}

	const x = [
		// Original 78 general causal features
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

		// 62 new general causal features (CV-verified):
		// Async background jobs & underway tool families (14)
		facts.bgJobs.size,
		bgTaskRunning,
		bgBashRunning,
		bgSlowRunning,
		Math.log(oldestBgAgeS + 10),
		facts.bgDoneCount,
		Math.log(avgBgDurS + 10),
		Math.log(facts.bgDoneMaxS + 10),
		Math.log1p(facts.waitCount),
		Math.log(facts.waitTimeSumS + 5),
		Math.log1p(facts.taskDispatches),
		underwayExecCount,
		Math.log(underwayExecAgeS + 5),
		underwayWaitOrTask,

		// Tool intent semantics & edit-verify stage progression (17)
		Math.log1p(facts.intentTestCount),
		Math.log1p(facts.intentE2eCount),
		Math.log1p(facts.intentBuildCount),
		Math.log1p(facts.intentGitCount),
		facts.intentReleaseCount > 0 ? 1 : 0,
		facts.lastIntentTestAtS >= 0 ? Math.log(Math.max(0, elapsedS - facts.lastIntentTestAtS) + 10) : logElapsed,
		facts.lastIntentGitAtS >= 0 ? Math.log(Math.max(0, elapsedS - facts.lastIntentGitAtS) + 10) : logElapsed,
		facts.underwayIntentTest ? 1 : 0,
		facts.underwayIntentE2e ? 1 : 0,
		facts.underwayIntentGit ? 1 : 0,
		facts.curStage,
		Math.log(Math.max(0, elapsedS - facts.stageAtS) + 10),
		Math.log1p(facts.stageTransitions),
		facts.editVerifyCycles,
		Math.log1p(facts.verificationsSeen),
		facts.lastErrorAtS >= 0 ? Math.log(Math.max(0, elapsedS - facts.lastErrorAtS) + 10) : logElapsed,
		Math.log1p(facts.subsequentToolsAfterLastWrite),

		// Working set / file dynamics (6)
		Math.log1p(nRead),
		Math.log1p(nEdit),
		reEditRatio,
		readNovelty,
		facts.editedTestFile ? 1 : 0,
		facts.editedDocFile ? 1 : 0,

		// Adaptive unshrunk todo velocity & scope (6)
		rawTodoLogRem,
		medTodoLogRem,
		rawTodoLogRem - logElapsed,
		curTodoExcess,
		todoScopeGrowth,
		remVerifyShare,

		// Context growth, LLM thinking latency, tool fanout & prose (9)
		Math.log1p(deltaCtx),
		Math.log1p(deltaCtx / Math.max(1, elapsedS)),
		facts.compactions,
		Math.log(avgThinkS + 2),
		Math.log(recThinkS + 2),
		avgFanout,
		Math.log1p(recOutAvg),
		facts.proseHasRemaining ? 1 : 0,
		facts.proseHasDebug ? 1 : 0,

		// Prompt structure, session history & half-prompt velocity trend (10)
		facts.promptIsQuestion ? 1 : 0,
		facts.promptIsTinyCmd ? 1 : 0,
		facts.promptHasCodeOrLog ? 1 : 0,
		Math.log1p(facts.promptFilePaths),
		facts.promptKwE2eBench ? 1 : 0,
		facts.promptKwCommitShip ? 1 : 0,
		Math.log(facts.gapSincePrevPromptS + 10),
		Math.log1p(facts.carryoverOpenTodos),
		Math.log(facts.priorMaxDurS + SMOOTH_S),
		toolHalfRatio,

		// Markov tool-transition & debug-loop features (5)
		Math.log1p(facts.writeToErrCount),
		Math.log1p(facts.errToWriteCount),
		Math.log1p(facts.okToTodoDoneCount),
		Math.log1p(facts.okToGitCount),
		Math.log1p(facts.debugLoops),

		// 4-channel 40-concept Ridge duration priors (5)
		zp,
		zp - logElapsed,
		zt,
		zi,
		zr,
	];

	return { x, elapsedS, logElapsed, anchorZ, hasActiveTodo, todoDoneThisPrompt, toolCalls: tools };
}
