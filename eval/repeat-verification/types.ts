export type Kind = string;
export interface TimingPrefix {
  call:number; kind:Kind; ageS:number; background:boolean;
  exact:number[]; family:number[]; priorPromptExact:number; priorPromptFamily:number;
}
export interface Frame {elapsedS:number; timing:TimingPrefix|null; toolLeftS:number|null}
export interface Span {
  kind:Kind; seconds:number; completed:boolean; background:boolean; noticeOnly:boolean;
  exactAnchorS:number|null; familyAnchorS:number|null;
}
export interface Replay {prompts:Frame[][]; spans:Span[]}
export interface ScoredPoint {
  local:number; exact:number; pooled:number; remaining:number; slow:boolean; timing:TimingPrefix|null;
  tool:{population:number; exact:number; pooled:number; gold:number|null; used:'exact'|'family'|null}|null;
}
/** duration is the completed prompt's active seconds, never a predictor input. */
export interface ScoredPrompt {session:string; duration:number; points:ScoredPoint[]}
