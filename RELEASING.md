# Releasing

> **Reality check (2026-06-23):** the OIDC `publish.yml` workflow has **never** succeeded — every
> release (v0.13 … v0.26) shows a failed *Publish to PyPI* run with `invalid-publisher` because the
> PyPI **trusted publisher was never registered** (the one-time step below is still pending). Every
> version on PyPI was uploaded by the **manual token path** in §"Cutting a release". That path is the
> source of truth today; the trusted-publishing path is the *intended* end state once the one-time
> setup is done — at which point the manual `uv publish` step can be dropped.

## Make the workflow work (one-time, maintainer, ~2 minutes) — then drop the token

Register the trusted publisher so the GitHub-Release workflow stops failing and CI can publish without
a stored token:

1. Log in to PyPI → the **causal-worlds** project → **Manage** → **Publishing** → **Add a trusted
   publisher** (https://pypi.org/manage/project/causal-worlds/settings/publishing/). (The project now
   exists, so this is a *normal* publisher, not a *pending* one.)
2. Fill in **exactly** these claims (they must match what the OIDC token presents — verified from a
   failed run's debug claims):
   - **Owner:** `noumenal-ai`
   - **Repository name:** `causal-worlds`
   - **Workflow name:** `publish.yml`
   - **Environment:** *(leave blank — the workflow sets none; `environment` is `MISSING` in the token)*
3. Save. The next GitHub Release will publish via OIDC; then delete the manual `uv publish` step here
   and the `PYPI_TOKEN` from `.env`.

## Cutting a release (the path that actually works today)

1. Bump `src/causal_worlds/_version.py` (the single source of truth; the wheel reads it).
2. Add a `CHANGELOG.md` entry under the new version.
3. Gate green: `make validate` (ruff `select=ALL` · mypy `strict` · pytest with the coverage floor).
4. Build + verify the artifact:
   ```bash
   uv build
   uvx twine check dist/*   # validates metadata + README rendering on PyPI
   ```
5. Commit (conventional, atomic), then tag and push:
   ```bash
   git tag -a vX.Y.Z <release-commit> -m "causal-worlds vX.Y.Z — <summary>"
   git push origin main vX.Y.Z
   ```
6. **Publish to PyPI (manual, until the trusted publisher is set up):**
   ```bash
   set -a; source .env; set +a               # loads PYPI_TOKEN (gitignored, never committed)
   uv publish --token "$PYPI_TOKEN" dist/causal_worlds-X.Y.Z*
   ```
7. Create the GitHub Release for the tag (`gh release create vX.Y.Z …`) for the human-readable notes.
   This *also* fires `publish.yml`, which will **fail** until the one-time setup above is done — that
   failure is currently expected and harmless (the manual upload in step 6 is what publishes).
8. Verify it's live (note: the package requires **Python ≥ 3.13** — test in a 3.13 env, and the
   resolver index can lag a minute behind the upload):
   ```bash
   uv venv --python 3.13 /tmp/cw-check
   VIRTUAL_ENV=/tmp/cw-check uv pip install "causal-worlds==X.Y.Z"
   ```
