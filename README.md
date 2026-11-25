# dsh-nvrun

[![CI](https://github.com/notxart/dsh-nvrun/actions/workflows/ci.yml/badge.svg)](https://github.com/notxart/dsh-nvrun/actions/workflows/ci.yml) [![Tag Version](https://img.shields.io/github/v/tag/notxart/dsh-nvrun?sort=semver&label=tag)](https://github.com/notxart/dsh-nvrun/tags)

**Linux only.** A compact NVIDIA CUDA runner for the DeepSeek Harness local
sandbox.

The bundle uses the official `@deepseek-ai/dsh-sandbox-local` `runnerCommand`
seam. It validates the exact bwrap profile DSH passes in, adds only an
allowlisted set of NVIDIA/WSL device nodes, and executes the profile through
bubblewrap or the official Landlock launcher. It does not replace the sandbox
provider or grant filesystem access.

## Why this exists

DSH's local sandbox has no GPU device policy:

- Landlock grants writes only to `/dev/null`, `/tmp`, and the workspace.
  CUDA needs a read-write open of `/dev/dxg` on WSL2.
- bwrap creates a fresh `/dev` and binds no host GPU nodes.

The result is `torch.cuda.is_available() == False`, or
`Failed to initialize NVML: GPU access blocked by the operating system`, even
when the same command works outside the sandbox.

This is the failure tracked in
[deepseek-harness discussion #2983](https://github.com/deepseek-ai/deepseek-harness/discussions/2983).

## Requirements

- Linux (including WSL2) on x86_64 or arm64.
- DSH `0.2.2-alpha.2` through `0.2.x`. The package declares
  `engines.dsh: ">=0.2.2-alpha.2 <0.3.0"`.
- Node.js `24.9.0` or newer. The package declares this as `engines.node`.
- NVIDIA driver and device nodes visible to the DSH **host process**:
  `nvidia-smi -L` must work outside the sandbox.
- One working runner:
  - **bubblewrap** (recommended), or
  - the official DSH `landlock-run` launcher. The package installs it as an
    optional dependency and the runner also checks `~/.local/bin`. The optional
    native payload is small and may duplicate a copy already shipped by DSH.

<!-- x-release-please-start-version -->

## Install with an agent

Paste this prompt into a DSH agent:

```text
Install the DeepSeek Harness bundle dsh-nvrun@1.0.1.

Hard constraints:
1. Use only the plugin_manager tool with action "install_bundle" and
   target "dsh-nvrun@1.0.1". This exact npm package and version is
   the only allowed install source.
2. Do not clone a repository, download a tarball, install from a local path,
   edit package.json or cordis.patch.yml, or install any other package.
3. The installed package must resolve inside the active DSH profile at
   $DSH_PROFILE_DIR/node_modules/dsh-nvrun. Do not move or copy it.
4. Stop and report the error if the registry package is unavailable, if the
   result is not application: "applied", or if there are warnings.
5. After installation, run only these read-only checks and do not request
   danger-full-access:

   nvidia-smi -L
   python3 -c "import ctypes; lib=ctypes.CDLL('libcuda.so.1'); assert lib.cuInit(0)==0; n=ctypes.c_int(); assert lib.cuDeviceGetCount(ctypes.byref(n))==0; print(n.value)"

Report the installed version and the exact GPU output.
```

The prompt deliberately names an exact package and version. A prompt such as
"install the GPU plugin" would let an agent choose an arbitrary source path or
package; do not use one.

## Manual install

From a DSH session:

```json
{
  "action": "install_bundle",
  "target": "dsh-nvrun@1.0.1"
}
```

A successful result contains `"application": "applied"`. If the result says
`"restart-required"`, restart DSH before testing.

<!-- x-release-please-end -->

## Runner selection

| Order | Runner | Source |
| --- | --- | --- |
| 1 | bwrap | `DSH_GPU_BWRAP`, then `/usr/bin/bwrap`, `/bin/bwrap`, `/usr/local/bin/bwrap`, `~/.local/bin/bwrap` |
| 2 | Landlock | `DSH_GPU_LANDLOCK_RUN`, then the DSH native package, `~/.local/bin/landlock-run`, `/usr/local/bin`, `/usr/bin`, `/bin` |

`DSH_GPU_RUNNER` can force `auto` (default), `bwrap`, or `landlock`. A forced
runner that is missing fails closed.

The Landlock path requires full enforcement. A partial-ABI kernel is rejected
because `runnerCommand` asserts full enforcement to DSH.

## Verify

In a session whose file policy is `workspace-write`:

```bash
nvidia-smi -L
python3 - <<'PY'
import ctypes

lib = ctypes.CDLL("libcuda.so.1")
assert lib.cuInit(0) == 0
count = ctypes.c_int()
assert lib.cuDeviceGetCount(ctypes.byref(count)) == 0
print("CUDA devices:", count.value)
PY
```

With uv, use a cache inside the writable workspace. The default
`~/.cache/uv` is outside the sandbox's writable paths and is read-only there;
uv fails with `Could not acquire lock` if you use it directly.

One-off check:

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
uv run --no-project --with 'jax[cuda13]' \
  python3 -c 'import jax; print(jax.devices())'
```

For a reusable environment:

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
uv venv .venv
uv pip install --python .venv/bin/python 'jax[cuda13]'
.venv/bin/python -c 'import jax; print(jax.devices())'
```

The plugin is active when DSH is applying a confined mode (`read-only` or
`workspace-write`). Under `danger-full-access`, DSH does not call the sandbox
runner at all.

The workspace profile must provide a writable temp directory for XLA. DSH's
`workspace-write` profile mounts an ephemeral `/tmp` under bwrap, so the default
profile is sufficient. Under Landlock, `/tmp` is the host `/tmp`; the wrapper
cannot create a private mount namespace, so Landlock users should treat that
temporary directory as shared host state.

## Security model

- Grants only `/dev/dxg`, `/dev/nvidia*`, `/dev/dri`, and `/dev/nvidia-caps`.
- Rejects symlinked device candidates. Landlock receives individual character
  devices, not the whole `/dev` tree.
- Strictly parses the provider's profile. Unknown flags, mismatched binds, a
  missing separator, or a missing command stop the command before it runs.
- Passes argv directly to the runner; no shell is involved and no device path
  comes from model output.
- Resolves executables from fixed locations or an explicit absolute override;
  a runner inside the workspace is rejected.
- No `danger-full-access` fallback. A missing runner or partial Landlock
  enforcement fails closed.
- Before the caller command, the wrapper runs a harmless probe (`true`, or a
  Node fallback on distributions without it) through the selected runner using
  the same profile and devices. A probe failure is re-emitted as
  `dsh-gpu-runner:` so DSH classifies it as a runner failure.
- Under `read-only`, the filesystem stays read-only, but the GPU character
  devices are granted read/write because CUDA/NVML need `O_RDWR` and ioctls.
  This is device access, not general filesystem write access.
- There is no model-facing tool and no install hook. The bundle only changes the
  sandbox runner configuration.

The host process and a hostile root are outside this threat model. This is not
a VM boundary.

## Troubleshooting

### `dsh-gpu-runner: NO_RUNNER`

Install bubblewrap, or set `DSH_GPU_LANDLOCK_RUN` to the absolute path of the
DSH launcher. Check `<launcher> --probe` prints `landlock: fully enforced`.

### `dsh-gpu-runner: LANDLOCK_UNAVAILABLE`

Landlock is not fully enforced on this kernel, or its preflight could not
install the ruleset. Use bwrap. Under Landlock, `/tmp` is host `/tmp`, not an
ephemeral bwrap mount.

### `dsh-gpu-runner: BWRAP_UNAVAILABLE`

The bwrap preflight could not create the namespace or bind profile. The user
command was not executed. Check `bwrap --version`, user namespaces, and the
profile paths.

### `dsh-gpu-runner: PROFILE_FLAG`

The sandbox provider added a profile flag this version does not recognize.
Update the plugin; do not make the runner ignore the flag.

### `GPU access blocked by the operating system`

The runner is active but the device grant is missing or ineffective. Check:

```bash
ls -l /dev/dxg /dev/dri /dev/nvidia* 2>&1
nvidia-smi -L
```

Fix the host/container GPU passthrough first if the nodes are absent.

### The plugin is installed but GPU access is unchanged

The session is probably in `danger-full-access`, where no sandbox runner runs.
Switch the session to `workspace-write`.

## Development

```bash
pnpm install
pnpm check
```

`pnpm check` runs:

1. `biome check` — formatter and recommended linter.
2. `tsc -p jsconfig.json` — JSDoc type checking with `noEmit`.
3. The `node:test` suite.

The runner is one JavaScript file, `runner.js`, with JSDoc type annotations.
`pnpm format` writes the configured style; `pnpm lint` only runs the linter.
TypeScript is a **check-only** dev tool here: `pnpm typecheck` runs `tsc` with
`noEmit`; it never creates `dist/` or ships to users. The only optional runtime
dependency is the official native Landlock launcher; bwrap users do not need
it.

### Local development install

For a source checkout, use the single resolved absolute path instead of the
registry package; never leave a variable or a search step in the prompt. The
runtime file is already `runner.js`; no build step is required.

```json
{
  "action": "install_bundle",
  "target": "/path/to/dsh-nvrun"
}
```

A path install uses the checkout as the package source. Do not delete or rename
the checkout while it is installed. For normal use, prefer the registry package
so the profile does not depend on a working tree.

## License

MIT.
