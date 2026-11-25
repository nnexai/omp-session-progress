export type Step = 'edit' | 'verify' | 'delegate' | 'release' | 'inspect' | 'report' | 'other';
export type Tail = 'final' | 'more' | 'unknown';
export interface Route { steps: Step[]; tail: Tail; conditional: boolean }
export interface Plans { prose: Route | null; checklist: Route | null }
