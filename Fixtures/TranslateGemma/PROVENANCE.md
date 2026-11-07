# Fixtures/TranslateGemma — provenance

Verbatim copy of one file from `mlx-community/translategemma-4b-it-4bit` @ `5788ec08c047f3f2e17808101b8d9566ac930d58`
(fetched 2026-11-04 with `hf download mlx-community/translategemma-4b-it-4bit --revision 5788ec08c047f3f2e17808101b8d9566ac930d58
--local-dir /private/tmp/translategemma`). It is the tiny stand-in for the 2.2 GB checkpoint, so the MC2-3 table and
template tests can run without installing the model. **Never edit it to make a test pass**; re-fetch from the pinned
revision instead.

| File | Why a test reads it | sha256 |
|---|---|---|
| `chat_template.jinja` | the language table is generated from its `languages` dict; swift-jinja renders it for three reference requests; the template's `{{ bos_token }}` is where the single BOS comes from (MC2 Traps 2–4) | `5e85514635b80a62b64c9982810bfdfbdd482be9a59ed12c25ded2bf3aabddaf` |

Used by `MLXUITests/TranslateGemmaTemplateTests.swift` (MC2-3), `MLXUITests/TranslateGemmaModuleTests.swift` (MC2-4) and
`MLXUITests/TranslateGemmaRealLoadTests.swift` (MC2-4's gated real-weights suite). Not part of the CatFlow fixture
manifest.

This repo's `tokenizer_config.json` has **no** `chat_template` key — the `.jinja` file is the template's only copy, and
swift-transformers picks a `chat_template.jinja` up from the model folder (`Hub.loadConfig`). The installer already
fetches it (`ModelFileSelector.metadataNames`), and `TranslateGemmaCatalogTests.theInstallerFetchesEveryFileTheEngineNeeds`
pins that.

## Generated files

**Not** verbatim copies — regenerate them with their own scripts from the repo root; the files record their provenance in
their headers. Never edit them to make a test pass: regenerate, and if they change that is a finding.

- `prompt_goldens.json` — `python3 -I Fixtures/TranslateGemma/make_prompt_goldens.py`. The pinned template rendered by
  Python's reference Jinja implementation (`ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)`, the
  same two options swift-transformers compiles with) for three requests: EN→MS, ZH-Hans→EN, and the regional
  `en_GB` case whose underscore the template turns into a hyphen. Python 3.13.3 / jinja2 3.1.6; `transformers` 5.12.0's
  `apply_chat_template(tokenize=False, add_generation_prompt=True)` was compared byte for byte and agrees.
  `TranslateGemmaTemplateTests.rendersThePinnedGoldensByteForByte` asserts swift-jinja's output is identical.
- `MLXUI/Modules/TranslateGemma/TranslateGemmaLanguages+Data.swift` — `python3 -I Fixtures/TranslateGemma/make_languages.py`.
  The template's `languages` dict as Swift: 581 codes → 161 display names, with the one `Chinese` name split into
  `Chinese (Simplified)` → `zh-Hans` and `Chinese (Traditional)` → `zh-Hant` (owner decision 6), 162 entries in template
  order. `TranslateGemmaTemplateTests.theTableIsExactlyTheTemplatesLanguages` parses the pinned template again and
  compares both directions.
