# Releasing VLMBox

VLMBox releases are published from version tags. The release workflow builds the wheel and source distribution once, publishes those exact files to TestPyPI, verifies that the wheel can be downloaded and that its metadata is correct, then waits at the protected `pypi` GitHub Environment before publishing to PyPI.

## One-time GitHub setup

1. Ensure https://github.com/czxvan/VLMBox is available and the intended release commit is pushed to its `main` branch.
2. In GitHub repository settings, create two Environments:
   - `testpypi`
   - `pypi`
3. Add `czxvan` as a required reviewer to the `pypi` Environment. Because this is currently a single-maintainer repository, `prevent_self_review` is disabled so the maintainer can approve their own release deployment. If a second maintainer is added later, enable `prevent_self_review` and add that maintainer as a reviewer.
4. Create or sign in to a separate TestPyPI account at https://test.pypi.org/.
5. Configure a Trusted Publisher on TestPyPI at https://test.pypi.org/manage/account/publishing/:
   - Owner: `czxvan`
   - Repository: `VLMBox`
   - Workflow: `publish.yml`
   - Environment: `testpypi`
6. Configure a second Trusted Publisher on PyPI at https://pypi.org/manage/account/publishing/ with the same owner, repository, and workflow, and Environment `pypi`.

The workflow uses OIDC Trusted Publishing. Do not create or store long-lived PyPI API tokens in GitHub Secrets.

## Release procedure

1. Ensure the intended release commit is on `main`, CI is green, and `project.version` in `pyproject.toml` is the intended version.
2. Create and push an annotated version tag, for example:

   ```bash
   git tag -a v0.1.0 -m "VLMBox 0.1.0"
   git push origin v0.1.0
   ```

3. The workflow builds and validates the wheel and source distribution, then publishes them to TestPyPI.
4. The workflow downloads the published TestPyPI wheel and verifies its archive and core metadata.
5. Before approving the `pypi` Environment, manually test installation from TestPyPI in a clean environment:

   ```bash
   python -m pip install \
     --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     "vlmbox==0.1.0"
   python -c "import vlmbox; print(vlmbox.__version__)"
   ```

   The extra index is needed because TestPyPI does not mirror the project's runtime dependencies. Use this command only for testing this known package name; do not use extra indexes as a general dependency-resolution policy.

6. Review the TestPyPI installation and the GitHub Actions run. Approve the pending `pypi` Environment deployment to publish the exact same distributions to PyPI.

## Important

- A version already uploaded to PyPI cannot be overwritten. If a release has a packaging defect, fix it and increment the version.
- TestPyPI and PyPI are separate indexes and each needs its own Trusted Publisher configuration.
- The workflow cannot publish until the GitHub repository and both Trusted Publishers are configured.
- The public source repository is https://github.com/czxvan/VLMBox. The PyPI Trusted Publisher must match owner `czxvan`, repository `VLMBox`, workflow `publish.yml`, and the exact environment name.
