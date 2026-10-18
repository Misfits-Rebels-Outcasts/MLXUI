---
description: Onboard a new MLX model — check mlx-swift support, then expand the RSI backlog with catalog + Run-UI + AI Workflows + Local Server wiring tasks.
argument-hint: <mlx-community/repo>  e.g. mlx-community/GLM-4.7-Flash-4bit
---

Run the Add-a-Model intake for the repo: **$ARGUMENTS**

Read `RSI/prompts/add-model.md` and follow it exactly, treating `<REPO>` as
`$ARGUMENTS`. In brief:

1. **Check mlx-swift support** (Step A) — identify the architecture from the HuggingFace
   `config.json`, then confirm a runner exists in `mlx-swift-lm` (or an app-side port).
   If none is found, prompt me — I may still choose to proceed with a port cycle.
2. **Classify** SUPPORTED vs NEEDS-PORT (Step B).
3. **Expand the backlog** (Step C) — append a chained `### $ARGUMENTS` AM-task group under
   *## Add-a-model intake* in `RSI/backlog.md`. For a SUPPORTED model collapse the
   download/module/UI steps into a single verify+smoke task; for NEEDS-PORT emit the full
   port cycle. Either way, include **AM-W (Workflows wiring)** — the step that adds a
   `BridgeEntry` to `CatalogBridge.swift`, the display name to `TaskModels.taskModels`,
   and a curated manifest JSON under `Resources/CatFlow/models/`, so the model is
   available in `Automate → AI Workflows`. And include **AM-S (Local Server readiness)**
   for LLM chat models — classify the family in `ToolCallFormatTable`,
   `ToolCallHistoryTable` and `ReasoningTemplateTable`, confirm `ServeMemoryAdmission` and
   `ModelContextWindow` can size it from `config.json`, decide the prompt cache, and add
   SV smoke rows — so the model serves correctly over `127.0.0.1:1212`. Other modalities
   record AM-S as skipped, pending S5.
4. **Prompt to start the RSI loop** (AM6) — present the group and ask before implementing.
   Do not begin coding without my go-ahead (L2 autonomy: one task per cycle,
   scorecard-first review).

Obey `RSI/policies.md`.
