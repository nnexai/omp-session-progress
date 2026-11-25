#!/usr/bin/env python3
"""Local Linux research CLI. Nothing is captured unless invoked in an opted-in session."""
import argparse
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
import tempfile
from time import monotonic_ns as now


def dump(path, value):
    # Publish complete records only; a killed process leaves a missing end, not a zero duration.
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(value, f, separators=(',', ':'))
    os.replace(f.name, path)


def boot():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def init():
    root = Path(tempfile.mkdtemp(prefix='omp-slow-checks-', dir='/tmp'))
    dump(root/'capture', {'version': 1, 'boot': boot(), 'salt': secrets.token_hex(32)})
    return root


def open_capture(root):
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('expected an owner-only capture directory')
    config = json.loads((root/'capture').read_bytes())
    if config['version'] != 1 or config['boot'] != boot():
        raise ValueError('different boot or unsupported capture format')
    return bytes.fromhex(config['salt'])


def command_id(salt, argv):
    # Filesystem bytes preserve the exact argv boundary; never write commands, cwd or output.
    return hmac.new(salt, b'\0'.join(map(os.fsencode, [os.getcwd(), *argv])), hashlib.sha256).hexdigest()


def run_command(argv):
    if not argv:
        raise ValueError('missing command')
    return subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, check=False).returncode


def previous(root, key, cutoff):
    # Only already-published successful exits count; ignore in-flight peers.
    newest, prior = -1, None
    for path in root.glob('check-*/end'):
        end = json.loads(path.read_bytes())
        if end['code'] or not newest < end['ns'] <= cutoff:
            continue
        start = json.loads((path.parent/'start').read_bytes())
        if start['key'] == key:
            newest, prior = end['ns'], end['ns'] - start['ns']
    return prior


def record(root, salt, number, argv):
    if number < 1:
        raise ValueError('check number must be positive')
    d = root/f'check-{number}'
    d.mkdir(mode=0o700)  # Single writer; never overwrite an earlier trial.
    with (d/'running').open('xb') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        key = command_id(salt, argv)
        prior = previous(root, key, now())
        t = now()
        dump(d/'start', {'ns': t, 'key': key, 'prior_ns': prior})
        code = run_command(argv)
        dump(d/'end', {'ns': now(), 'code': code})
    return code


def activity(root, label, start, end, number=None):
    dump(root/('span-'+secrets.token_hex(8)), {'kind': label, 'check': number, 'start': start, 'end': end})


def join_check(root, number):
    # This witnesses AGENT waiting only when the caller runs join synchronously,
    # not when a shell or the tool host backgrounds it. No inference from silence.
    d = root/f'check-{number}'
    if not (d/'start').is_file():
        raise ValueError('check has not been launched')
    with (d/'running').open('rb') as lock:
        started = now()
        try:
            fcntl.flock(lock, fcntl.LOCK_SH)
        finally:
            activity(root, 'join', started, now(), number)
    if not (d/'end').is_file():
        raise ValueError('check runner stopped without recording an exit')
    return json.loads((d/'end').read_bytes())['code']


def work(root, argv):
    t = now()
    try:
        return run_command(argv)
    finally:
        activity(root, 'other', t, now())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('init', help='opt into a new capture for this session; prints its private directory')
    for name in ['run', 'join', 'work', 'report']:
        p = sub.add_parser(name)
        p.add_argument('root', type=Path)
        if name in ('run','join'):
            p.add_argument('number', type=int)
        if name in ('run','work'):
            p.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == 'init':
        print(init()); return 0
    salt = open_capture(args.root)
    if args.action == 'report':
        from summary import report
        print(json.dumps(report(args.root), indent=2, allow_nan=False)); return 0
    if args.action == 'join':
        return join_check(args.root, args.number)
    cmd = args.command[1:] if args.command[:1] == ['--'] else args.command
    if args.action == 'run':
        return record(args.root, salt, args.number, cmd)
    return work(args.root, cmd)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, KeyError, ValueError) as e:
        # An OS exception can contain argv[0] or a private path. Do not echo it.
        print(f'Capture failed ({type(e).__name__}); check the private directory, command and run number.', file=sys.stderr)
        sys.exit(1)
