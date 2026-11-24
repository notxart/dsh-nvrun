import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  buildBwrapArgv,
  buildLandlockArgv,
  gpuEnv,
  isNvidiaDeviceName,
  listGpuDevices,
  parseProfile,
  runRunner,
} from '../runner.js';

const READ_ONLY = [
  '--ro-bind',
  '/',
  '/',
  '--dev',
  '/dev',
  '--unshare-pid',
  '--proc',
  '/proc',
  '--die-with-parent',
  '--',
  'bash',
  '-c',
  'true',
];

const WORKSPACE = [
  '--ro-bind',
  '/',
  '/',
  '--dev',
  '/dev',
  '--unshare-pid',
  '--proc',
  '/proc',
  '--die-with-parent',
  '--tmpfs',
  '/tmp',
  '--bind',
  '/workspace',
  '/workspace',
  '--',
  'bash',
  '-c',
  'true',
];

/** @type {{ path: string, kind: 'file' | 'directory' }[]} */
const DEVICES = [
  { path: '/dev/dxg', kind: 'file' },
  { path: '/dev/dri', kind: 'directory' },
];

test('parses read-only and workspace-write profiles', () => {
  const readOnly = parseProfile(READ_ONLY);
  assert.deepEqual(readOnly.readOnly, ['/']);
  assert.deepEqual(readOnly.readWrite, []);
  assert.deepEqual(readOnly.command, ['bash', '-c', 'true']);

  const workspace = parseProfile(WORKSPACE);
  assert.deepEqual(workspace.readOnly, ['/']);
  assert.deepEqual(workspace.readWrite, ['/tmp', '/workspace']);
});

test('rejects unknown flags, a missing separator, or a missing command', () => {
  assert.throws(
    () => parseProfile([...READ_ONLY.slice(0, 8), '--clearenv', '--', 'true']),
    /PROFILE_FLAG/,
  );
  assert.throws(() => parseProfile(READ_ONLY.slice(0, -4)), /PROFILE_SEPARATOR/);
  assert.throws(
    () => parseProfile(['--ro-bind', '/workspace/', '/workspace/', ...READ_ONLY.slice(3)]),
    /PROFILE_PATH/,
  );
  assert.throws(() => parseProfile([...READ_ONLY.slice(0, -4), '--']), /PROFILE_COMMAND/);
});

test('injects only allowlisted device binds after --dev /dev', () => {
  const argv = buildBwrapArgv(parseProfile(WORKSPACE), DEVICES);
  assert.deepEqual(argv.slice(5, 11), [
    '--dev-bind',
    '/dev/dxg',
    '/dev/dxg',
    '--dev-bind',
    '/dev/dri',
    '/dev/dri',
  ]);
  assert.deepEqual(argv.slice(-3), ['bash', '-c', 'true']);
});

test('translates the profile to Landlock grants', () => {
  const argv = buildLandlockArgv(parseProfile(WORKSPACE), [{ path: '/dev/dxg', kind: 'file' }]);
  assert.deepEqual(argv.slice(0, 10), [
    '--ro',
    '/',
    '--rw',
    '/dev/null',
    '--rw',
    '/tmp',
    '--rw',
    '/workspace',
    '--rw',
    '/dev/dxg',
  ]);
  assert.deepEqual(argv.slice(-3), ['bash', '-c', 'true']);
});

test('adds WSL lookup paths once and preserves existing values', () => {
  const env = gpuEnv({ PATH: '/usr/bin' }, { hasWslLib: true });
  assert.equal(env.PATH, '/usr/bin:/usr/lib/wsl/lib');
  assert.equal(env.LD_LIBRARY_PATH, '/usr/lib/wsl/lib');
  assert.equal(gpuEnv(env, { hasWslLib: true }).PATH, '/usr/bin:/usr/lib/wsl/lib');
  assert.equal(
    gpuEnv({}, { hasWslLib: true }).PATH,
    '/usr/local/bin:/usr/bin:/bin:/usr/lib/wsl/lib',
  );
});

test('matches native NVIDIA device names, including hyphenated UVM nodes', () => {
  [
    'nvidia0',
    'nvidiactl',
    'nvidia-uvm',
    'nvidia-uvm-tools',
    'nvidia-modeset',
    'nvidia-caps',
  ].forEach((name) => {
    assert.equal(isNvidiaDeviceName(name), true);
  });
  ['dxg', 'dri', 'nvidiacontrol', 'nvidia-uvm-evil'].forEach((name) => {
    assert.equal(isNvidiaDeviceName(name), false);
  });
});

test('lists only character devices and GPU directories', { skip: !existsSync('/dev') }, () => {
  const devices = listGpuDevices();
  devices.forEach((device) => {
    assert.ok(device.path.startsWith('/dev/'));
    assert.ok(device.kind === 'file' || device.kind === 'directory');
  });
});

test('passes the profile to bwrap and preserves its exit code', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-gpu-runner-'));
  try {
    const argsFile = join(dir, 'args.txt');
    const devRoot = join(dir, 'dev');
    const bwrap = join(dir, 'bwrap');
    mkdirSync(devRoot);
    writeFileSync(
      bwrap,
      '#!/bin/sh\nlast=\nfor arg in "$@"; do last="$arg"; done\nif [ "$last" = /bin/true ]; then exit 0; fi\nprintf \'%s\\n\' "$@" > "$GPUBWRAP_ARGS"\nexit 7\n',
    );
    chmodSync(bwrap, 0o755);
    const code = await runRunner(WORKSPACE, {
      cwd: process.cwd(),
      devRoot,
      env: {
        ...process.env,
        DSH_GPU_RUNNER: 'bwrap',
        DSH_GPU_BWRAP: bwrap,
        GPUBWRAP_ARGS: argsFile,
      },
    });
    assert.equal(code, 7);
    const args = readFileSync(argsFile, 'utf8').trim().split('\n');
    assert.deepEqual(args.slice(-3), ['bash', '-c', 'true']);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('classifies a bwrap startup failure as a runner failure', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-gpu-bwrap-fail-'));
  try {
    const bwrap = join(dir, 'bwrap');
    writeFileSync(bwrap, "#!/bin/sh\nprintf 'bwrap: simulated startup failure\\n' >&2\nexit 1\n");
    chmodSync(bwrap, 0o755);
    await assert.rejects(
      runRunner(READ_ONLY, {
        env: { ...process.env, DSH_GPU_RUNNER: 'bwrap', DSH_GPU_BWRAP: bwrap },
      }),
      /BWRAP_UNAVAILABLE/,
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('classifies a Landlock startup failure as a runner failure', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-gpu-landlock-fail-'));
  try {
    const landlock = join(dir, 'landlock-run');
    writeFileSync(
      landlock,
      "#!/bin/sh\nprintf 'landlock-run: simulated startup failure\\n' >&2\nexit 125\n",
    );
    chmodSync(landlock, 0o755);
    await assert.rejects(
      runRunner(READ_ONLY, {
        env: {
          ...process.env,
          DSH_GPU_RUNNER: 'landlock',
          DSH_GPU_LANDLOCK_RUN: landlock,
        },
      }),
      /LANDLOCK_UNAVAILABLE/,
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('CLI prefixes a bwrap startup failure for DSH classification', () => {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-gpu-cli-fail-'));
  try {
    const bwrap = join(dir, 'bwrap');
    writeFileSync(bwrap, "#!/bin/sh\nprintf 'bwrap: simulated startup failure\\n' >&2\nexit 1\n");
    chmodSync(bwrap, 0o755);
    const result = spawnSync(
      process.execPath,
      [join(import.meta.dirname, '..', 'runner.js'), ...READ_ONLY],
      {
        encoding: 'utf8',
        env: { ...process.env, DSH_GPU_RUNNER: 'bwrap', DSH_GPU_BWRAP: bwrap },
      },
    );
    assert.equal(result.status, 125);
    assert.match(result.stderr, /^dsh-gpu-runner: BWRAP_UNAVAILABLE:/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('fails closed when a requested bwrap is missing', async () => {
  await assert.rejects(
    runRunner(READ_ONLY, {
      env: { ...process.env, DSH_GPU_RUNNER: 'bwrap', DSH_GPU_BWRAP: '/does/not/exist' },
    }),
    /RUNNER_PATH/,
  );
});

test('runs through a symlinked CLI entry point', () => {
  const dir = mkdtempSync(join(tmpdir(), 'dsh-gpu-symlink-'));
  try {
    const bwrap = join(dir, 'bwrap');
    const link = join(dir, 'runner-link.js');
    writeFileSync(bwrap, '#!/bin/sh\nprintf SYMLINK_CLI\\n\nexit 0\n');
    chmodSync(bwrap, 0o755);
    symlinkSync(join(import.meta.dirname, '..', 'runner.js'), link);
    const profile = [...READ_ONLY.slice(0, -3), '/bin/echo', 'ignored'];
    const result = spawnSync(process.execPath, [link, ...profile], {
      encoding: 'utf8',
      env: { ...process.env, DSH_GPU_RUNNER: 'bwrap', DSH_GPU_BWRAP: bwrap },
    });
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /SYMLINK_CLI/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('rejects a runner inside a symlinked workspace', async () => {
  const root = mkdtempSync(join(tmpdir(), 'dsh-gpu-symlink-ws-'));
  try {
    const real = join(root, 'real');
    const alias = join(root, 'alias');
    const bwrap = join(real, 'bwrap');
    mkdirSync(real);
    symlinkSync(real, alias, 'dir');
    writeFileSync(bwrap, '#!/bin/sh\nexit 0\n');
    chmodSync(bwrap, 0o755);
    await assert.rejects(
      runRunner(READ_ONLY, {
        cwd: alias,
        env: { ...process.env, DSH_GPU_RUNNER: 'bwrap', DSH_GPU_BWRAP: bwrap },
      }),
      /RUNNER_UNTRUSTED/,
    );
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

const LANDLOCK = process.env.DSH_GPU_LANDLOCK_RUN ?? join(homedir(), '.local/bin/landlock-run');

test('runs through Landlock when a full launcher is available', {
  skip: !existsSync(LANDLOCK) || !existsSync('/dev/dxg'),
}, async () => {
  const code = await runRunner(READ_ONLY, {
    env: {
      ...process.env,
      DSH_GPU_RUNNER: 'landlock',
      DSH_GPU_LANDLOCK_RUN: LANDLOCK,
    },
  });
  assert.equal(code, 0);
});
