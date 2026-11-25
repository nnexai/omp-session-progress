import { expect, test } from 'bun:test';
import { runnerFamily } from './runner';

test('recognizes stable runner families through foreground wrappers',()=>{
  const cases:[string,string][]=[
    ['cd project && npm run test -- --filter unit','npm:test'],
    ['npx playwright test --project desktop','playwright:test'],
    ['cd .. && cd "some dir" && CI=1 RUST_BACKTRACE=1 cargo test --workspace 2>&1 | tail -n 40','cargo:test#workspace'],
    ['cargo test -p sample','cargo:test#target'],
    ['NODE_ENV=test npm run test:e2e > build.log','npm:test#e2e'],
    ['pnpm test:unit >> results.log 2>&1','pnpm:test#unit'],
    ['python3 -m pytest -n 2 tests/unit','pytest:test'],
    ['npm exec -- vitest run --project unit','vitest:test'],
    ['bunx tsc --noEmit','tsc:check'],
    ['ruff check src','ruff:check'],
    ['cargo clippy','cargo:lint'],
    ['go vet ./...','go:check'],
    ['make e2e','make:e2e'],
    ['yarn typecheck','yarn:check'],
    ['bun test','bun:test'],
    ['eslint src','eslint:lint'],
    ['jest tests','jest:test'],
    ['mypy src','mypy:check'],
  ];
  for (const [command,expected] of cases) expect(runnerFamily(command)).toBe(expected);
});

test('rejects opaque, asynchronous, non-executing, and unsafe commands',()=>{
  const commands=[
    'echo npm test', './script test', 'npm run verify', 'cargo test &',
    'cargo test && echo done', 'cd project; cargo test', 'cd $(pwd) && cargo test',
    'ENV=$(malicious) cargo test', 'cargo test `echo x`', 'cargo test > $(echo log)',
    'cargo test | head -n 10', 'cargo test | grep pass', 'cargo test | tail -n 4 &',
    'cargo test 2>&1 | tail -n 4 | head -n 1', 'cargo test > log; echo done',
    'cargo test < input', 'cargo test\necho done', 'sh -c "cargo test"',
    'npm exec npm exec npm exec cargo test', 'npx npm exec cargo test',
    'cargo test --list', 'cargo test -- --list', 'pytest --version',
    'pytest --collect-only', 'tsc --help', 'tsc --watch', 'ruff format src',
    'ruff --version', 'vitest watch', 'vitest list', 'jest --listTests',
    'go test -list .', 'playwright test --list', 'npm run test:watch',
    'constructor --list', 'toString',
    'make test --dry-run', 'cargo test >', 'cargo test || echo failed',
  ];
  for (const command of commands) expect(runnerFamily(command)).toBeNull();
});
