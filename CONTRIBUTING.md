# Contributing to MLXUI

Thanks for helping. MLXUI is a Mac app for running local AI models on Apple Silicon with MLX, and for chaining them into
workflows (CAT Flow).

## Adding a model

The most common contribution is a new MLX model. The whole procedure is in
**[docs/contributing/add-a-model.md](docs/contributing/add-a-model.md)**: support checks, the catalog entry, the module,
the Run screen, workflow wiring, tests and the PR description.

In short:

1. **Open an issue first** with the **Add a model** template, and wait for a maintainer to accept it.
2. **Post your intake** (Steps A–C of the guide) as a comment on the issue, and wait for the go-ahead.
3. **Implement** in a draft PR, one commit per task, and fill in the PR description as the guide asks.

**Using an AI agent?** Agents are welcome. Copy the ready-made prompt from
[Contributing with an AI agent](docs/contributing/add-a-model.md#contributing-with-an-ai-agent). It tells your agent which
files to read, where to stop and wait, and what never to change.

## Other contributions

Bug reports and fixes are welcome too. Open an issue describing what you saw, on which Mac and macOS, with steps to
reproduce, before sending a fix. Keep a change to the one thing it fixes. Read [CLAUDE.md](CLAUDE.md) for the architecture
and conventions; its references to `RSI/`, `Design/` and `scripts/` are private maintainer material that isn't in the repo.

## Ground rules

- Swift packages, `project.pbxproj` package wiring, entitlements, Keychain and network code change only with the owner's
  approval. Propose the change on an issue first.
- Extend tests; don't weaken or edit existing expectations, goldens or fixtures.
- Never commit personal data (recordings, photos, documents) as test inputs.
- An Apple Silicon Mac is needed to run models and the real-weights tests.
