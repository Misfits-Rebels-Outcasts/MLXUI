---
description: Onboard a new MLX model — check mlx-swift support, then expand the RSI backlog with catalog + Run-UI tasks.
argument-hint: <mlx-community/repo>  e.g. mlx-community/GLM-4.7-Flash-4bit
---

Run the Add-a-Model intake for the repo: **$ARGUMENTS**

Read `MLXUI/RSI/prompts/add-model.md` and follow it exactly, treating `<REPO>` as
`$ARGUMENTS`. In brief:

1. **Check mlx-swift support** (Step A) — identify the architecture from the HuggingFace
   `config.json`, then confirm a runner exists in `mlx-swift-lm` (or an app-side port).
   If none is found, prompt me — I may still choose to proceed with a port cycle.
2. **Classify** SUPPORTED vs NEEDS-PORT (Step B).
3. **Expand the backlog** (Step C) — append a chained `### $ARGUMENTS` AM-task group under
   *## Add-a-model intake* in `MLXUI/RSI/backlog.md`. For a SUPPORTED model collapse the
   download/module/UI steps into a single verify+smoke task; for NEEDS-PORT emit the full
   port cycle.
4. **Prompt to start the RSI loop** (AM6) — present the group and ask before implementing.
   Do not begin coding without my go-ahead (L1 autonomy).

Obey `MLXUI/RSI/policies.md`.
