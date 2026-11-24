/**
 * GPU-aware runner for the official sandbox-local `runnerCommand` seam.
 *
 * The provider hands this process a bwrap-compatible profile. The runner
 * validates that profile, adds only the documented NVIDIA/WSL device nodes,
 * and executes it through bwrap or the official Landlock launcher.
 */

import { spawn, spawnSync } from 'node:child_process';
import {
  accessSync,
  constants,
  existsSync,
  lstatSync,
  readdirSync,
  realpathSync,
  statSync,
} from 'node:fs';
import { homedir, constants as osConstants } from 'node:os';
import { isAbsolute, join, normalize, sep } from 'node:path';
import { pathToFileURL } from 'node:url';

const FAILURE_EXIT = 125;
const PROBE_TIMEOUT_MS = 3000;
const NVIDIA_NODE = /^nvidia(?:[0-9]+|ctl|-?uvm(?:-tools)?|-?modeset|-?caps)$/;
/** @type {NodeJS.Signals[]} */
const FORWARDED_SIGNALS = ['SIGINT', 'SIGTERM', 'SIGHUP', 'SIGQUIT'];
const GPU_DIRECTORIES = new Set(['dri', 'nvidia-caps']);
const WSL_LIB = '/usr/lib/wsl/lib';

/**
 * @typedef {object} Profile
 * @property {readonly string[]} bwrapArgs
 * @property {readonly string[]} command
 * @property {readonly string[]} readOnly
 * @property {readonly string[]} readWrite
 */

/**
 * @typedef {object} Device
 * @property {string} path
 * @property {'file' | 'directory'} kind
 */

/**
 * @typedef {object} RunOptions
 * @property {string} [cwd]
 * @property {string} [devRoot]
 * @property {Record<string, string | undefined>} [env]
 * @property {string} [platform]
 */

export class RunnerError extends Error {
  /**
   * @param {string} code
   * @param {string} message
   */
  constructor(code, message) {
    super(`${code}: ${message}`);
    this.name = 'RunnerError';
    /** @type {string} */
    this.code = code;
  }
}

/**
 * @param {unknown} error
 * @returns {boolean}
 */
function isMissing(error) {
  return error instanceof Error && 'code' in error && error.code === 'ENOENT';
}

/**
 * Treat optional GPU device entries as absent when the caller cannot inspect
 * them. This never grants access; it prevents optional DRM/UVM entries from
 * breaking every sandboxed command.
 * @param {unknown} error
 * @returns {boolean}
 */
function isIgnorableFsError(error) {
  return (
    error instanceof Error &&
    'code' in error &&
    (error.code === 'ENOENT' || error.code === 'EACCES' || error.code === 'EPERM')
  );
}

/**
 * @param {string | undefined} value
 * @param {string} label
 * @returns {string}
 */
function assertAbsolute(value, label) {
  const trailing = value !== undefined && value.length > 1 && value.endsWith(sep);
  if (value === undefined || !isAbsolute(value) || normalize(value) !== value || trailing) {
    throw new RunnerError('PROFILE_PATH', `${label} must be a normalized absolute path`);
  }
  return value;
}

/**
 * @param {readonly string[]} values
 * @returns {string[]}
 */
function unique(values) {
  return [...new Set(values)];
}

/**
 * @param {string} parent
 * @param {string} child
 * @returns {boolean}
 */
function isWithin(parent, child) {
  /**
   * @param {string} value
   * @returns {string}
   */
  const canonical = (value) => {
    const normalized = normalize(value);
    return normalized.length > 1 ? normalized.replace(/\/+$/u, '') : normalized;
  };
  const base = canonical(parent);
  const target = canonical(child);
  return target === base || target.startsWith(base === sep ? sep : `${base}${sep}`);
}

/**
 * Resolve a path while tolerating a path that disappeared during discovery.
 * @param {string} target
 * @returns {string}
 */
function safeRealpath(target) {
  try {
    return realpathSync(target);
  } catch {
    return normalize(target);
  }
}

/**
 * Parse the exact bwrap-compatible profile emitted by sandbox-local.
 * @param {readonly string[]} argv
 * @returns {Profile}
 */
export function parseProfile(argv) {
  /** @type {string[]} */
  const readOnly = [];
  /** @type {string[]} */
  const readWrite = [];
  let hasDev = false;
  let hasPid = false;
  let hasProc = false;
  let hasDieWithParent = false;
  let separator = -1;
  let index = 0;

  while (index < argv.length) {
    const token = argv[index];
    if (token === undefined || token.length === 0) {
      throw new RunnerError('PROFILE_ARGV', 'profile arguments must be non-empty strings');
    }
    if (token === '--') {
      separator = index;
      break;
    }
    if (token === '--ro-bind' || token === '--bind') {
      const source = assertAbsolute(argv[index + 1], `${token} source`);
      const target = assertAbsolute(argv[index + 2], `${token} target`);
      if (source !== target) {
        throw new RunnerError('PROFILE_BIND', `${token} source and target must match`);
      }
      (token === '--ro-bind' ? readOnly : readWrite).push(source);
      index += 3;
    } else if (token === '--dev') {
      if (argv[index + 1] !== '/dev') {
        throw new RunnerError('PROFILE_DEV', '--dev must target /dev');
      }
      hasDev = true;
      index += 2;
    } else if (token === '--proc') {
      if (argv[index + 1] !== '/proc') {
        throw new RunnerError('PROFILE_PROC', '--proc must target /proc');
      }
      hasProc = true;
      index += 2;
    } else if (token === '--tmpfs') {
      if (argv[index + 1] !== '/tmp') {
        throw new RunnerError('PROFILE_TMPFS', '--tmpfs must target /tmp');
      }
      readWrite.push('/tmp');
      index += 2;
    } else if (token === '--unshare-pid') {
      hasPid = true;
      index += 1;
    } else if (token === '--die-with-parent') {
      hasDieWithParent = true;
      index += 1;
    } else {
      throw new RunnerError(
        'PROFILE_FLAG',
        `unsupported sandbox profile flag ${JSON.stringify(token)}`,
      );
    }
  }

  if (separator < 0) {
    throw new RunnerError('PROFILE_SEPARATOR', 'sandbox profile has no -- separator');
  }
  const command = argv.slice(separator + 1);
  if (command.length === 0) {
    throw new RunnerError('PROFILE_COMMAND', 'sandbox profile has no wrapped command');
  }
  if (!hasDev || !hasPid || !hasProc || !hasDieWithParent) {
    throw new RunnerError(
      'PROFILE_INCOMPLETE',
      'sandbox profile is missing a required isolation flag',
    );
  }
  if (!readOnly.includes('/')) {
    throw new RunnerError('PROFILE_ROOT', 'sandbox profile must read-only bind /');
  }

  return {
    bwrapArgs: argv.slice(0, separator),
    command,
    readOnly: unique(readOnly),
    readWrite: unique(readWrite),
  };
}

/**
 * Match a GPU device name, including native Linux hyphenated names.
 * @param {string} name
 * @returns {boolean}
 */
export function isNvidiaDeviceName(name) {
  return NVIDIA_NODE.test(name);
}

/**
 * Enumerate only the fixed WSL/NVIDIA device allowlist. Symlinks are ignored.
 * @param {string} [devRoot]
 * @returns {Device[]}
 */
export function listGpuDevices(devRoot = '/dev') {
  const names = new Set(['dxg', 'dri', 'nvidia-caps']);
  try {
    readdirSync(devRoot).forEach((name) => {
      if (isNvidiaDeviceName(name)) names.add(name);
    });
  } catch (error) {
    if (isMissing(error)) return [];
    throw new RunnerError('GPU_STAT', `cannot inspect ${devRoot}`);
  }

  /** @type {Device[]} */
  const devices = [];
  [...names].sort().forEach((name) => {
    const path = join(devRoot, name);
    let stats;
    try {
      stats = lstatSync(path);
    } catch (error) {
      if (isIgnorableFsError(error)) return;
      throw new RunnerError('GPU_STAT', `cannot inspect ${path}`);
    }
    if (stats.isSymbolicLink()) return;
    if (GPU_DIRECTORIES.has(name) && stats.isDirectory()) {
      devices.push({ path, kind: 'directory' });
      return;
    }
    if (stats.isCharacterDevice()) devices.push({ path, kind: 'file' });
  });
  return devices;
}

/**
 * @param {readonly Device[]} devices
 * @returns {string[]}
 */
function landlockDeviceGrants(devices) {
  const grants = devices.flatMap((device) => {
    if (device.kind === 'file') return [device.path];
    let entries;
    try {
      entries = readdirSync(device.path);
    } catch (error) {
      if (isIgnorableFsError(error)) return [];
      throw new RunnerError('GPU_DIRECTORY', `cannot enumerate ${device.path}`);
    }
    return entries.sort().flatMap((entry) => {
      const path = join(device.path, entry);
      try {
        const stats = lstatSync(path);
        return !stats.isSymbolicLink() && stats.isCharacterDevice() ? [path] : [];
      } catch (error) {
        if (isIgnorableFsError(error)) return [];
        throw new RunnerError('GPU_STAT', `cannot inspect ${path}`);
      }
    });
  });
  return unique(grants);
}

/**
 * Inject `--dev-bind` immediately after the provider's `--dev /dev`.
 * @param {Profile} profile
 * @param {readonly Device[]} devices
 * @returns {string[]}
 */
export function buildBwrapArgv(profile, devices) {
  const at = profile.bwrapArgs.findIndex(
    (token, index) => token === '--dev' && profile.bwrapArgs[index + 1] === '/dev',
  );
  if (at < 0) throw new RunnerError('PROFILE_DEV', 'sandbox profile has no --dev /dev');
  const binds = devices.flatMap((device) => ['--dev-bind', device.path, device.path]);
  return [
    ...profile.bwrapArgs.slice(0, at + 2),
    ...binds,
    ...profile.bwrapArgs.slice(at + 2),
    '--',
    ...profile.command,
  ];
}

/**
 * Translate the same profile to the official Landlock launcher grammar.
 * @param {Profile} profile
 * @param {readonly Device[]} devices
 * @returns {string[]}
 */
export function buildLandlockArgv(profile, devices) {
  const readOnly = unique(['/', ...profile.readOnly]);
  const readWrite = unique(['/dev/null', ...profile.readWrite, ...landlockDeviceGrants(devices)]);
  return [
    ...readOnly.flatMap((path) => ['--ro', path]),
    ...readWrite.flatMap((path) => ['--rw', path]),
    '--',
    ...profile.command,
  ];
}

/**
 * Add WSL driver lookup paths once; preserve caller values.
 * @param {Record<string, string | undefined>} [env]
 * @param {{ wslLib?: string, hasWslLib?: boolean }} [options]
 * @returns {Record<string, string | undefined>}
 */
export function gpuEnv(env = process.env, options = {}) {
  const wslLib = options.wslLib ?? WSL_LIB;
  if (!(options.hasWslLib ?? existsSync(wslLib))) return { ...env };
  const next = { ...env };
  /**
   * @param {'PATH' | 'LD_LIBRARY_PATH'} name
   * @param {string} [fallback]
   * @returns {void}
   */
  const append = (name, fallback) => {
    const current = next[name];
    if (current === undefined || current === '') {
      next[name] = fallback === undefined ? wslLib : `${fallback}:${wslLib}`;
    } else if (!current.split(':').includes(wslLib)) {
      next[name] = `${current}:${wslLib}`;
    }
  };
  /**
   * @param {string} name
   * @returns {void}
   */
  const prepend = (name) => {
    const current = next[name];
    if (current === undefined || current === '') {
      next[name] = wslLib;
    } else if (!current.split(':').includes(wslLib)) {
      next[name] = `${wslLib}:${current}`;
    }
  };
  prepend('LD_LIBRARY_PATH');
  append('PATH', '/usr/local/bin:/usr/bin:/bin');
  return next;
}

/**
 * @param {string} name
 * @param {string} selected
 * @param {string} cwd
 * @returns {string}
 */
function validateExecutable(name, selected, cwd) {
  let real;
  try {
    if (!statSync(selected).isFile()) throw new Error('not a regular file');
    accessSync(selected, constants.X_OK);
    real = realpathSync(selected);
  } catch {
    throw new RunnerError('RUNNER_PATH', `${name} is not an executable regular file: ${selected}`);
  }
  const realCwd = safeRealpath(cwd);
  if (
    isWithin(cwd, real) ||
    isWithin(cwd, selected) ||
    isWithin(realCwd, real) ||
    isWithin(realCwd, selected)
  ) {
    throw new RunnerError(
      'RUNNER_UNTRUSTED',
      `${name} must not live inside the workspace: ${selected}`,
    );
  }
  return real;
}

/**
 * @param {string} name
 * @param {string} envName
 * @param {readonly string[]} candidates
 * @param {Record<string, string | undefined>} env
 * @param {string} cwd
 * @returns {string | undefined}
 */
function resolveExecutable(name, envName, candidates, env, cwd) {
  const configured = env[envName];
  if (configured !== undefined) {
    if (configured.length === 0 || !isAbsolute(configured)) {
      throw new RunnerError('RUNNER_PATH', `${envName} must be an absolute path`);
    }
    return validateExecutable(name, configured, cwd);
  }
  for (const candidate of candidates) {
    if (!existsSync(candidate)) continue;
    try {
      return validateExecutable(name, candidate, cwd);
    } catch (error) {
      if (error instanceof RunnerError && error.code === 'RUNNER_UNTRUSTED') throw error;
    }
  }
  return undefined;
}

/**
 * @returns {Promise<string | undefined>}
 */
async function packagedLandlock() {
  try {
    const specifier = '@deepseek-ai/node-addon-system/landlock-run';
    const module = /** @type {{ launcherPath?: () => string }} */ (await import(specifier));
    const path = module.launcherPath?.();
    return path !== undefined && existsSync(path) ? path : undefined;
  } catch {
    return undefined;
  }
}

/**
 * @param {Record<string, string | undefined>} env
 * @param {string} cwd
 * @returns {string | undefined}
 */
function findBwrap(env, cwd) {
  return resolveExecutable(
    'bubblewrap',
    'DSH_GPU_BWRAP',
    ['/usr/bin/bwrap', '/bin/bwrap', '/usr/local/bin/bwrap', join(homedir(), '.local/bin/bwrap')],
    env,
    cwd,
  );
}

/**
 * @param {Record<string, string | undefined>} env
 * @param {string} cwd
 * @returns {Promise<string | undefined>}
 */
async function findLandlock(env, cwd) {
  const packaged = await packagedLandlock();
  const candidates = [
    ...(packaged === undefined ? [] : [packaged]),
    join(homedir(), '.local/bin/landlock-run'),
    '/usr/local/bin/landlock-run',
    '/usr/bin/landlock-run',
    '/bin/landlock-run',
  ];
  return resolveExecutable('landlock-run', 'DSH_GPU_LANDLOCK_RUN', candidates, env, cwd);
}

/**
 * Extract one bounded stderr line from a failed preflight.
 * @param {{ error?: Error, status: number | null, stderr?: string }} result
 * @returns {string}
 */
function preflightDetail(result) {
  const line = String(result.stderr ?? '')
    .split(/\r?\n/)
    .map((value) => value.trim())
    .find((value) => value.length > 0);
  return line ?? `exit status ${result.status ?? 'signal'}`;
}

/**
 * Pick a harmless executable that exists in the sandbox root. Prefer the
 * conventional true binaries, then fall back to the running Node executable
 * for distributions without /bin/true.
 * @returns {string[]}
 */
function probeCommand() {
  for (const candidate of ['/bin/true', '/usr/bin/true']) {
    if (existsSync(candidate)) return [candidate];
  }
  return [process.execPath, '-e', 'process.exit(0)'];
}

/**
 * Prove that the selected initiator can create its confinement and execute a
 * harmless command before the caller's command is started. Any failure here
 * is a wrapper-owned failure, so DSH can classify it from the dsh-gpu-runner
 * prefix instead of treating it as a normal command failure.
 * @param {'BWRAP_UNAVAILABLE' | 'LANDLOCK_UNAVAILABLE'} code
 * @param {string} name
 * @param {string} runner
 * @param {readonly string[]} args
 * @param {Record<string, string | undefined>} env
 * @param {string} cwd
 * @returns {{ stderr: string }}
 */
function preflightRunner(code, name, runner, args, env, cwd) {
  const result = spawnSync(runner, args, {
    cwd,
    encoding: 'utf8',
    env,
    stdio: ['ignore', 'ignore', 'pipe'],
    timeout: PROBE_TIMEOUT_MS,
  });
  const stderr = String(result.stderr ?? '');
  if (result.error !== undefined || result.status !== 0) {
    throw new RunnerError(
      code,
      `${name} failed before executing the command: ${preflightDetail(result)}`,
    );
  }
  return { stderr };
}

/**
 * @param {string} command
 * @param {readonly string[]} args
 * @param {Record<string, string | undefined>} env
 * @param {string} cwd
 * @returns {Promise<number>}
 */
function spawnRunner(command, args, env, cwd) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, env, stdio: 'inherit' });
    let settled = false;
    /**
     * @param {NodeJS.Signals} signal
     * @returns {void}
     */
    const forward = (signal) => {
      if (child.exitCode === null && child.signalCode === null) child.kill(signal);
    };
    const listeners = FORWARDED_SIGNALS.map((signal) => {
      const handler = () => forward(signal);
      process.on(signal, handler);
      return { signal, handler };
    });
    const cleanup = () => {
      listeners.forEach(({ signal, handler }) => {
        process.off(signal, handler);
      });
    };
    /**
     * @param {number | null} code
     * @param {NodeJS.Signals | null} signal
     * @returns {void}
     */
    const finish = (code, signal) => {
      if (settled) return;
      settled = true;
      cleanup();
      if (code !== null) {
        resolve(code);
        return;
      }
      resolve(128 + (signal === null ? 1 : (osConstants.signals[signal] ?? 1)));
    };
    child.once('error', (error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(error);
    });
    child.once('close', finish);
  });
}

/**
 * Execute one wrapped command with the selected real sandbox runner.
 * @param {readonly string[]} argv
 * @param {RunOptions} [options]
 * @returns {Promise<number>}
 */
export async function runRunner(argv, options = {}) {
  const env = options.env ?? process.env;
  const cwd = options.cwd ?? process.cwd();
  const platform = options.platform ?? process.platform;
  if (platform !== 'linux') {
    throw new RunnerError(
      'PLATFORM',
      `GPU sandbox grants are supported on Linux/WSL2 only, not ${platform}`,
    );
  }

  const profile = parseProfile(argv);
  const devices = listGpuDevices(options.devRoot);
  const childEnv = gpuEnv(env);
  const mode = env.DSH_GPU_RUNNER ?? 'auto';
  if (mode !== 'auto' && mode !== 'bwrap' && mode !== 'landlock') {
    throw new RunnerError('RUNNER_MODE', 'DSH_GPU_RUNNER must be auto, bwrap, or landlock');
  }

  const bwrap = mode === 'landlock' ? undefined : findBwrap(env, cwd);
  if (bwrap !== undefined) {
    const args = buildBwrapArgv(profile, devices);
    const probeArgs = buildBwrapArgv({ ...profile, command: probeCommand() }, devices);
    preflightRunner('BWRAP_UNAVAILABLE', 'bubblewrap', bwrap, probeArgs, childEnv, cwd);
    return spawnRunner(bwrap, args, childEnv, cwd);
  }
  if (mode === 'bwrap') {
    throw new RunnerError('RUNNER_UNAVAILABLE', 'bubblewrap was requested but not found');
  }

  const landlock = await findLandlock(env, cwd);
  if (landlock === undefined) {
    throw new RunnerError(
      'NO_RUNNER',
      'no bubblewrap or Landlock launcher found; install bubblewrap or set DSH_GPU_LANDLOCK_RUN',
    );
  }
  const args = buildLandlockArgv(profile, devices);
  const probeArgs = buildLandlockArgv({ ...profile, command: probeCommand() }, devices);
  const preflight = preflightRunner(
    'LANDLOCK_UNAVAILABLE',
    'landlock-run',
    landlock,
    probeArgs,
    childEnv,
    cwd,
  );
  if (preflight.stderr.includes('partially')) {
    throw new RunnerError(
      'LANDLOCK_PARTIAL',
      'landlock reported partial enforcement during preflight',
    );
  }
  return spawnRunner(landlock, args, childEnv, cwd);
}

/**
 * @param {unknown} error
 * @returns {string}
 */
function diagnostic(error) {
  const message = error instanceof Error ? error.message : String(error);
  return `dsh-gpu-runner: ${message.replaceAll(/\s+/g, ' ').trim().slice(0, 1500)}`;
}

/**
 * CLI entry: one-line diagnostics and a fail-closed exit code.
 * @param {readonly string[]} argv
 * @returns {Promise<number>}
 */
export async function main(argv) {
  try {
    return await runRunner(argv);
  } catch (error) {
    process.stderr.write(`${diagnostic(error)}\n`);
    return FAILURE_EXIT;
  }
}

/**
 * Check whether this module was invoked directly as the CLI entry script,
 * resolving symlinks created by pnpm or local bundle installs.
 * @param {string | undefined} argv1
 * @returns {boolean}
 */
function isCliEntry(argv1) {
  if (argv1 === undefined) return false;
  try {
    return import.meta.url === pathToFileURL(realpathSync(argv1)).href;
  } catch {
    return import.meta.url === pathToFileURL(argv1).href;
  }
}

if (isCliEntry(process.argv[1])) {
  process.exitCode = await main(process.argv.slice(2));
}
