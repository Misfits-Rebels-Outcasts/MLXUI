---
description: Onboard a new MLX model — check mlx-swift support, classify it, then plan and implement the catalog, Run UI, AI Workflows and Local Server wiring.
argument-hint: <hf-repo>  e.g. mlx-community/GLM-4.7-Flash-4bit
---

Run the Add-a-Model intake for the repo: **$ARGUMENTS**

**Pick the procedure first. Check which file exists:**

- **Maintainers:** if `RSI/prompts/add-model.md` exists in this checkout, read it and follow it exactly, treating `<REPO>`
  as `$ARGUMENTS`. Obey `RSI/policies.md`. It appends the AM-task group to `RSI/backlog.md`, uses the automated smoke
  harness (`RSI/evals/model-smoke.sh`), and ends by asking whether to start the RSI loop. Don't begin coding without that
  go-ahead (L2: one task per cycle, scorecard-first review).
- **Contributors** (no `RSI/` folder, which is the case for any clone from GitHub): read
  `docs/contributing/add-a-model.md` and follow it, treating `<repo>` as `$ARGUMENTS`. In brief:
  1. **Find the GitHub issue for this model** and read it in full; its decisions override the guide. If there's no issue,
     stop and tell the user to open one first (the guide says what it should contain).
  2. **Step A:** read the repo's `config.json`; find the runner in the package checkouts **at the pinned revisions**
     (`Package.resolved`; read sources under DerivedData), never GitHub `main`; list the files the loader needs, including
     anything fetched from a second repo; size, fit and license.
  3. **Step B:** classify as SUPPORTED / SUPPORTED-in-package / NEEDS-MODULE / NEEDS-PORT, citing file:line. A port needs
     maintainer approval.
  4. **Step C:** draft the intake comment in the guide's format and show it to the user to post on the issue. **Stop
     there.** Implement (Step D) only after a maintainer has replied with a go-ahead, and the user says so.
  5. **Step D:** one commit per task, a draft PR early, both schemes built after every task, gated real-weights tests,
     and the PR description in the guide's six-part shape. Never change packages, `project.pbxproj` package wiring,
     entitlements, Keychain or network code, existing tests, fixtures, gallery flows or existing manifests.

If neither file exists, stop and say so. Don't improvise the procedure.
