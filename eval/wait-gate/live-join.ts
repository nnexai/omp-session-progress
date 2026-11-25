// Linux research smoke: this entire command must be the agent's synchronous
// foreground operation. A backgrounded copy proves only that a helper waited.
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { Readable } from 'node:stream';
import { Observer } from '../../src/observer';
import { extractSessionText } from '../extract';
import { ReplayClock } from '../tool-timing/replay';
import { forecast } from './forecast';

const [root,number,out]=process.argv.slice(2), file=process.env.PI_SESSION_FILE;
if (!root || !number || !out || !file) throw Error('Expected CAPTURE_DIRECTORY CHECK_NUMBER NEW_RESULT_JSON in the opted-in session');
// Freeze the real current-session prefix before joining. During the synchronous
// call the same Observer keeps ticking; this is a research reading, not a widget.
const trace=extractSessionText(readFileSync(file,'utf8'),file);
const clock=new ReplayClock(), observer=new Observer(clock);
observer.enable('wait-gate',()=>{});
for (const {t,event} of trace.events) {clock.advance(t,observer);observer.observe('wait-gate',event);}
if (!observer.report().facts.running) throw Error('No active prompt in the authorized prefix');
const local=()=>{clock.advance(Date.now(),observer);const min=observer.report().reading?.estimate.remainingMinutes;return min==null?null:min*60;};
const state={entry:0,blocked:0,released:0}, counts={waiting:0,changedWhileWaiting:0,notWaiting:0,changedOutsideWait:0,abstained:0};
let active=false, prior:number|null=null, initialAge=0, received=0, tick:ReturnType<typeof setInterval>|undefined;
let entryCheckLeftS:number|null=null,entryPromptS:number|null=null,entryGatedS:number|null=null;
function sample(){
  const baseline=local(); if(baseline===null){counts.abstained++;return null;}
  const age=initialAge+Math.max(0,performance.now()-received)/1000;
  const v=forecast(baseline,prior!==null&&prior>=60?prior:null,age,active);
  if(active){counts.waiting++;counts.changedWhileWaiting+=Number(v.local!==v.gated);}
  else {counts.notWaiting++;counts.changedOutsideWait+=Number(v.local!==v.gated);}
  return v;
}
const child=Bun.spawn(['python3','-B',resolve(import.meta.dir,'../slow-checks/capture.py'),'join',root,number,'--events'],{stdout:'pipe',stderr:'ignore'});
try {
  const lines=createInterface({input:Readable.fromWeb(child.stdout),crlfDelay:Infinity});
  for await(const line of lines){
    const e=JSON.parse(line) as {state:keyof typeof state;priorS:number|null;ageS:number};
    if (!Object.hasOwn(state,e.state) || state[e.state] || state.released || (e.state!=='entry' && !state.entry)) throw Error('Unexpected join event order');
    state[e.state]++;
    initialAge=e.ageS;prior=e.priorS;received=performance.now();active=e.state==='blocked';
    clearInterval(tick);
    const v=sample();
    if(active){entryCheckLeftS=v?.left??null;entryPromptS=v?.local??null;entryGatedS=v?.gated??null;tick=setInterval(sample,15000);}
  }
  if(await child.exited!==0||!state.entry||!state.released||counts.changedOutsideWait)throw Error('No confirmed successful join or invalid gate');
  writeFileSync(out,JSON.stringify({method:'Live targeted synchronous join, shipped Observer seeded from current session prefix; no whole-prompt label at this point.',...state,...counts,entryCheckLeftS,entryPromptS,entryGatedS},null,2)+'\n',{mode:0o600,flag:'wx'});
} finally {active=false;clearInterval(tick);observer.disable();}
