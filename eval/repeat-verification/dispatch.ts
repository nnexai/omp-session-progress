import { forecast, type Model } from './model';
import type { Span } from './types';

// Distinct from the display test: each finished call votes once, including
// those so fast they never intersect a 5/15-second Observer checkpoint.
export function dispatchReport(model:Model,test:Span[][]){
  const rows=test.flatMap((spans,session)=>spans.map(span=>({span,session})).filter(r=>r.span.completed).map(({span:s,session})=>{
    const pred=forecast(model,s.kind,0,s.exactAnchorS,s.familyAnchorS);
    return {session,exact:s.exactAnchorS!==null,prior:s.familyAnchorS!==null,truth:s.seconds/60,pred};
  }));
  function slice(group:typeof rows){const valid=group.filter(r=>r.pred?.used),mae=(arm:'population'|'pooled')=>valid.length?valid.reduce((n,r)=>n+Math.abs(r.pred![arm]-r.truth),0)/valid.length:null;
    return {calls:group.length,matchedCalls:valid.length,matchedSessions:new Set(valid.map(r=>r.session)).size,populationMae:mae('population'),historyMae:mae('pooled')};}
  const starts=test.flat();
  return {starts:starts.length,completed:rows.length,background:starts.filter(s=>s.background).length,
    terminalNotices:starts.filter(s=>s.noticeOnly).length,exact:slice(rows.filter(r=>r.exact)),
    familyOnly:slice(rows.filter(r=>!r.exact&&r.prior)),all:slice(rows),
    atLeastOneMinute:slice(rows.filter(r=>r.truth>=1)),atLeastTenMinutes:slice(rows.filter(r=>r.truth>=10))};
}
