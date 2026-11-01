---
name: Add a model
about: Propose a new MLX model for MLXUI (read docs/contributing/add-a-model.md first)
title: "Add <model name>: <what it does, in a few words>"
labels: ["model", "proposal"]
---

<!--
Thanks for proposing a model. Please read docs/contributing/add-a-model.md first.
A maintainer will accept the proposal (or not) and record any decisions here before work starts.
Using an AI agent? The guide has a ready-made prompt: docs/contributing/add-a-model.md#contributing-with-an-ai-agent
-->

## Model

- **Hugging Face repo:** <!-- e.g. mlx-community/Some-Model-4bit -->
- **Model card / upstream:** <!-- link -->
- **License (exactly as on the card):** <!-- e.g. Apache-2.0. Non-commercial licenses need a maintainer decision. -->
- **Download size:** <!-- sum of the weight files, GB -->

## What it adds

<!-- One paragraph: what can MLXUI do with this model that it can't do today? Which modality (chat, vision, OCR,
speech, audio, image, video, embedding, …)? Which AI Workflows row would use it, if any? -->

## Runner (if you've checked)

<!-- model_type / architectures from config.json, and where the runner lives in the pinned packages
(mlx-swift-lm, mlx-audio-swift) or "none found". "Not checked yet" is fine at this stage. -->

## You

- **Mac and RAM you'll test on:** <!-- e.g. M2 Pro, 32 GB -->
- **Plan to implement it yourself?** <!-- yes / no / with an AI agent (which one) -->

---

<!-- Maintainers: decisions, scope, and the work plan go below this line once the proposal is accepted. -->
