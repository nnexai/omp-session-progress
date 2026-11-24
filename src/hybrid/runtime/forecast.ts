import type { Evidence } from './evidence';
import { combine, enabled, instruction, packet, query, type Workload } from '../tune/policy';

export type Ask = (system:string, packet:string, signal:AbortSignal)=>Promise<{text:string; ms:number}>;
const labels:Record<Workload,true> = { CLOSE:true, BOUNDED:true, MULTI:true, EXTENDED:true, UNKNOWN:true };
interface Opinion { label:Workload; sourceS:number; readyAt:number }

/** One instance per prompt. The replay waits for I/O, but the displayed ETA only
 * incorporates a reply at the first checkpoint after its actual response time. */
export class HybridForecast {
  #calls=0;
  #lastS:number|null=null;
  #pending:Opinion|null=null;
  #current:Opinion|null=null;
  constructor(readonly mode:'policy'|'pilot') {}

  async sample(e:Evidence, atMs:number, ask:Ask, signal:AbortSignal) {
    if (!enabled && this.mode!=='pilot') return e.localMin;
    if (this.#pending && atMs>=this.#pending.readyAt) {
      this.#current=this.#pending;
      this.#pending=null;
    }
    // No external opinion can be used at the checkpoint that requested it.
    const current=this.#current;
    const predicted=current ? combine(e,current.label,Math.max(0,e.elapsedS-current.sourceS)) : e.localMin;
    if (!this.#pending && this.#calls<3 && query(e,this.#lastS)) {
      this.#calls++;
      this.#lastS=e.elapsedS;
      const {text,ms}=await ask(instruction,packet(e),signal);
      const label=text.trim().toUpperCase();
      this.#pending={label:Object.hasOwn(labels,label)?label as Workload:'UNKNOWN',sourceS:e.elapsedS,readyAt:atMs+ms};
    }
    return predicted;
  }
}
