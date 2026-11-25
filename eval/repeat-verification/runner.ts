// Only shell syntax whose foreground lifetime is apparent is accepted. Tokens are
// local to classification; neither arguments nor paths are returned as labels.
function tokensOf(command:string):string[]|null {
  if (/[\r\n`$;(){}\\#]/.test(command)) return null;
  const tokens:string[]=[];
  let i=0;
  while (i<command.length) {
    if (/\s/.test(command[i]!)) { i++; continue; }
    const operator=command.slice(i).match(/^(?:2>&1|&&|>>|>|\|)/)?.[0];
    if (operator) { tokens.push(operator); i+=operator.length; continue; }
    if (command[i]==='&' || command[i]==='<') return null;
    let word='';
    while (i<command.length && !/[\s&|<>]/.test(command[i]!)) {
      const c=command[i]!;
      if (c==='"' || c==="'") {
        const end=command.indexOf(c,i+1);
        if (end<0) return null;
        word+=command.slice(i+1,end);
        i=end+1;
      } else { word+=c; i++; }
    }
    if (!word) return null;
    tokens.push(word);
  }
  return tokens;
}

const nonExecution=/^(?:(?:--(?:help|version|list|list-tests|listTests|collect-only|collectonly|watch|watchAll|watch-all|showConfig|print-config))(?:=.*)?|-h|-V|-list)$/;
const assignment=/^[A-Za-z_][A-Za-z0-9_]*=/;

function classify(tokens:string[], nested=false):string|null {
  if (!tokens.length || tokens.some(t=>nonExecution.test(t))) return null;
  const exe=tokens[0]!.replace(/^.*\//,'');
  const args=tokens.slice(1);
  if ((exe==='vitest' && /^(?:list|watch|dev|bench|typecheck|init|related)$/.test(args[0]??'')) ||
      (exe==='tsc' && args.some(t=>t==='-w' || t==='--watch')) ||
      (exe==='pytest' && args.some(t=>t==='--setup-only'))) return null;
  const direct:Record<string,string>={pytest:'pytest:test',vitest:'vitest:test',jest:'jest:test',eslint:'eslint:lint',tsc:'tsc:check',mypy:'mypy:check'};
  if(Object.prototype.hasOwnProperty.call(direct,exe)) return direct[exe]!;
  if (exe==='ruff') return args[0]==='check' ? 'ruff:check' : null;
  if(exe==='cargo' && /^(test|check|clippy)$/.test(args[0]??'')) {
    const scope=args.includes('--workspace')||args.includes('--all')?'#workspace':args.some(a=>['-p','--package','--test','--lib','--bin'].includes(a))?'#target':'';
    return `cargo:${args[0]==='clippy'?'lint':args[0]}${scope}`;
  }
  if (exe==='go' && /^(test|vet)$/.test(args[0]??'') && !args.includes('-n')) return `go:${args[0]==='vet'?'check':'test'}`;
  if (exe==='playwright' && args[0]==='test') return 'playwright:test';
  if (exe==='make' && /^(test|check|lint|e2e)$/.test(args[0]??'') && !args.includes('-n') && !args.includes('--dry-run')) return `make:${args[0]}`;
  if (['npm','pnpm','yarn','bun'].includes(exe)) {
    if (args[0]==='exec') {
      if (nested) return null;
      const rest=args.slice(1);
      if (rest[0]==='--') rest.shift();
      return classify(rest,true);
    }
    let i=0;
    if (args[i]==='run' || args[i]==='run-script') i++;
    const action=args[i];
    const match=action?.match(/^(test|check|lint|e2e|typecheck)(?::(unit|e2e|integration))?$/);
    if(match) return `${exe}:${match[1]==='typecheck'?'check':match[1]}${match[2]?'#'+match[2]:''}`;
  }
  if (exe==='npx' || exe==='bunx') {
    if (nested) return null;
    const rest=args[0]==='--' ? args.slice(1) : args;
    return classify(rest,true);
  }
  if ((exe==='python' || exe==='python3') && args[0]==='-m' && args[1]==='pytest') return 'pytest:test';
  return null;
}

/** Classify only recognizable, foreground verification commands. */
export function runnerFamily(command:string):string|null {
  const tokens=tokensOf(command);
  if (!tokens) return null;
  let start=0;
  while (tokens[start]==='cd' && tokens[start+1] && tokens[start+2]==='&&') start+=3;
  while (tokens[start] && assignment.test(tokens[start]!)) start++;
  const remaining=tokens.slice(start);
  const pipe=remaining.indexOf('|');
  if (pipe>=0) {
    const tail=remaining.slice(pipe+1);
    if (tail.length!==3 || tail[0]!=='tail' || tail[1]!=='-n' || !/^\d+$/.test(tail[2]!)) return null;
    remaining.splice(pipe);
  }
  // Redirects are allowed only after the runner's arguments, not as shell programs.
  const redirect=remaining.findIndex(t=>t==='>' || t==='>>' || t==='2>&1');
  if (redirect>=0) {
    let i=redirect;
    while (i<remaining.length) {
      if (remaining[i]==='2>&1') { i++; continue; }
      if ((remaining[i]==='>' || remaining[i]==='>>') && remaining[i+1] && !/^(?:>|>>|2>&1|&&|\|)$/.test(remaining[i+1]!)) { i+=2; continue; }
      return null;
    }
    remaining.splice(redirect);
  }
  if (remaining.some(t=>t==='&&' || t==='|' || t==='>' || t==='>>' || t==='2>&1')) return null;
  return classify(remaining);
}
