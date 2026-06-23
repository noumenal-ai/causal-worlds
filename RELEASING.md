# Releasing

Releases publish to [PyPI](https://pypi.org/) automatically via **Trusted Publishing** (OIDC) — no
API token is ever stored in the repo or CI.

## One-time PyPI setup (maintainer, ~2 minutes)

Because `causal-worlds` is not yet on PyPI, register a **pending** trusted publisher:

1. Log in to PyPI → **Your projects** → **Publishing** → **Add a pending publisher**
   (https://pypi.org/manage/account/publishing/).
2. Fill in:
   - **PyPI project name:** `causal-worlds`
   - **Owner:** `noumenal-ai`
   - **Repository name:** `causal-worlds`
   - **Workflow name:** `publish.yml`
   - **Environment:** *(leave blank)*
3. Save. The first GitHub Release will then create the project and publish it.

(After the first publish it becomes a normal trusted publisher — no further setup.)

## Cutting a release

1. Bump `src/causal_worlds/_version.py` (the single source of truth; the wheel reads it).
2. Add a `CHANGELOG.md` entry under the new version.
3. Gate green: `make validate` (ruff `select=ALL` · mypy `strict` · pytest with the coverage floor).
4. Commit (conventional, atomic), then tag and push:
   ```bash
   git tag -a vX.Y.Z -m "causal-worlds vX.Y.Z — <summary>"
   git push origin main vX.Y.Z
   ```
5. Create the GitHub Release for the tag (`gh release create vX.Y.Z ...`). Publishing it triggers
   [`.github/workflows/publish.yml`](.github/workflows/publish.yml), which builds and uploads to PyPI.

## Verify locally before tagging

```bash
uv build
uvx twine check dist/*   # validates metadata + README rendering on PyPI
```
