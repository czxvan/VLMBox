# Releasing VLMBox

VLMBox releases are published directly to PyPI from version tags. The workflow runs lint and unit tests, builds the wheel and source distribution once, validates their metadata, and publishes those exact artifacts to PyPI using OIDC Trusted Publishing.

## One-time setup

1. Ensure the repository is https://github.com/czxvan/VLMBox and the intended release commit is pushed to `main`.
2. Keep a GitHub Environment named `pypi` in repository settings. It must have **no required reviewers or other protection rules**; the environment name is part of the PyPI Trusted Publisher identity.
3. On the PyPI account-level Publishing page (https://pypi.org/manage/account/publishing/), configure a Trusted Publisher with:
   - PyPI project name: `vlmbox`
   - Owner: `czxvan`
   - Repository: `VLMBox`
   - Workflow filename: `publish.yml`
   - Environment: `pypi`

The workflow uses OIDC Trusted Publishing. Do not create or store long-lived PyPI API tokens in GitHub Secrets.

## Release procedure

1. Ensure the intended release commit is on `main`, CI is green, and `project.version` in `pyproject.toml` is the intended version.
2. Create and push an annotated version tag, for example:

   ```bash
   git tag -a v0.1.1 -m "VLMBox 0.1.1"
   git push origin v0.1.1
   ```

3. The tag triggers the Publish workflow. It checks that the tag matches the project version, runs Ruff and unit tests, builds the wheel and source distribution, and validates the distributions with Twine.
4. The workflow publishes the validated artifacts directly to PyPI.
5. Verify the release:

   ```bash
   uv pip install --upgrade vlmbox
   uv pip show vlmbox
   ```

## Important

- A version already uploaded to PyPI cannot be overwritten. If a release has a packaging defect, fix it and increment the version.
- The PyPI Trusted Publisher must match owner `czxvan`, repository `VLMBox`, workflow `publish.yml`, and Environment `pypi`.
- Keep the `pypi` GitHub Environment, but do not add required reviewers or wait timers; doing so would block automatic publishing.
