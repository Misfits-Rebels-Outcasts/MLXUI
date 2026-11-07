# Q3 — Text encoder: exact tensor, prompt format, and whether mlx-swift-lm can produce it

**Short answer.** ACE-Step 1.5 uses `Qwen3-Embedding-0.6B` as a **plain causal `Qwen3Model`
forward** and takes its **final `last_hidden_state`** (post final RMSNorm, `[B, L, 1024]`, no
pooling, no LM head) over a **SFT-formatted caption prompt truncated to 256 tokens**. The lyric
side does *not* go through the transformer at all — it is a raw `embed_tokens` lookup. **A public
MLXEmbedders API returns exactly that tensor** (`Qwen3Model.callAsFunction(...).hiddenStates`),
and the app **already calls that same prototype method** in `EmbeddingEngine`. Recommendation:
**use the API — do not copy a forward pass.** Details and the two caveats below.

---

## 1. What upstream takes from `Qwen3-Embedding-0.6B`

### 1.1 The model object

`acestep/core/generation/handler/init_service_loader_components.py:93-103`:

```python
text_encoder_path = os.path.join(checkpoint_dir, "Qwen3-Embedding-0.6B")
self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
self.text_encoder  = AutoModel.from_pretrained(text_encoder_path)
self.text_encoder  = self.text_encoder.to(device).to(self.dtype)
```

- `AutoModel` + `config.architectures: ["Qwen3Model"]` + `model_type: qwen3` →
  **`transformers.Qwen3Model`**, i.e. the *bare trunk*: `embed_tokens` + 28 decoder layers +
  final `norm`. No `lm_head`, no pooler, no classifier. This is confirmed by the checkpoint
  itself: 310 tensors, no `lm_head` (`findings/weights.md` §2).
- **dtype:** the service dtype is `torch.float32` on every non-CUDA/non-XPU device —
  `init_service_orchestrator.py:101`: `self.dtype = torch.bfloat16 if resolved_device == "xpu"
  else torch.float32`. On Apple Silicon (`mps`) the text encoder therefore **runs in f32** even
  though the on-disk weights are bf16 (it is upcast at load). The MLX port may run bf16 for speed,
  but f32 is the parity target.
- **device:** the resolved runtime device (`mps` on Apple Silicon), or CPU when
  `offload_to_cpu` (`:97-101`).

### 1.2 The exact tensor: `last_hidden_state`

`acestep/core/generation/handler/conditioning_embed.py:71-79`:

```python
def infer_text_embeddings(self, text_token_idss):
    with torch.inference_mode():
        return self.text_encoder(input_ids=text_token_idss, lyric_attention_mask=None).last_hidden_state

def infer_lyric_embeddings(self, lyric_token_ids):
    with torch.inference_mode():
        return self.text_encoder.embed_tokens(lyric_token_ids)
```

- **Text path → `last_hidden_state`**, i.e. `norm(h)` after layer 27 — **not** an intermediate
  layer, **not** a pooled vector, **not** mean/CLS. Because `Qwen3Model.forward` applies
  `self.norm` before returning `last_hidden_state`, the final RMSNorm **is** part of the tensor.
  A port that skips the final norm will be silently wrong.
- **Attention mask is NOT passed to the text encoder.** The forward call gets `input_ids` only
  (the `lyric_attention_mask=None` kwarg is a naming accident of the shared helper; the model
  ignores it). The HF attention mask that *is* computed
  (`conditioning_text.py:132-133`) is used **only downstream**, to mask cross-attention keys in
  the conditioner. For a right-padded batch this is equivalent to running each prompt
  unpadded — which is what a Swift single-prompt path should do.
- **Lyric path → `embed_tokens` only.** `lyric_token_ids` never enter the transformer. The
  `[L, 151669→1024]` embedding table is looked up, giving `[B, L, 1024]`, which the conditioner
  then projects with its own `lyric_encoder.embed_tokens` (`Linear(1024, 2048)`) before its 8
  transformer layers (`modeling_acestep_v15_turbo.py:1525` and
  `AceStepLyricEncoder.forward` at `modeling_acestep_v15_turbo.py:606`, projection at `:626`).
  **So a Qwen3 forward is needed once per request, for the caption prompt only.**

### 1.3 Prompt / instruction format

Built in `conditioning_text.py:116` (and `prompt_utils.py:99`) from the template in
`acestep/constants.py:160-167`:

```
SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""
```

with the instruction defaulting to `DEFAULT_DIT_INSTRUCTION` (`constants.py:119`):

```
"Fill the audio semantic mask based on the given conditions:"
```

so a full text-encoder input is:

```
# Instruction
Fill the audio semantic mask based on the given conditions:

# Caption
<caption>

# Metas
<metas dict / JSON-ish string><|endoftext|>
```

Key points:

- **The tokenizer's chat template is NOT used.** `AutoTokenizer(...)` is called on a plain Python
  string with `padding="longest", truncation=True, max_length=256, return_tensors="pt"`
  (`conditioning_text.py:127-133`, `prompt_utils.py:111-117`). The checkpoint ships a
  `chat_template.jinja`, and the tokenizer config has `chat_template: None` — there is no
  `apply_chat_template` anywhere in the inference path (only in the vendored
  `third_parts/nano-vllm` example). **No special tokens are added**
  (`add_bos_token: false`, `add_eos_token` unset), so the sequence is the raw BPE of the string —
  the `<|endoftext|>` is literally part of the template text.
- Lyric prompt, if it were tokenized through the transformer (it is not — but the string is fed
  to `embed_tokens` after tokenization): `# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>`
  (`prompt_utils.py:27-29`), `max_length=2048` (`conditioning_text.py:142`).
- The `# Metas` value is a serialized dict string; for a port, this is whatever the caller fills
  in (BPM/key/duration/etc.). Format is identical for the SFT and turbo checkpoints.

### 1.4 Max sequence length at inference

- **Text: `max_length = 256`**, truncation on, `padding="longest"` — `conditioning_text.py:131`
  and `:175`, `prompt_utils.py:114`.
- **Lyrics: `max_length = 2048`** (`conditioning_text.py:142`) — but again this only bounds the
  `embed_tokens` lookup.
- The model *supports* 32 768 positions (`max_position_embeddings`); 256 is the **inference**
  bound. A Swift port should hard-code 256 for the text trunk and 2048 for the lyric token list.
- Because 256 ≪ 32768, RoPE tables can be built for 256 positions only.

### 1.5 How the result is projected into the condition encoder

`AceStepConditionEncoder` (`modeling_acestep_v15_turbo.py:1513-1525`, HF `:1506-1518`):

```python
self.text_projector = nn.Linear(config.text_hidden_dim, config.hidden_size, bias=False)  # 1024 → 2048
```

and `forward` (`:1544`) does `text_hidden_states = self.text_projector(text_hidden_states)`.
So: **text_hidden_dim 1024 → hidden_size 2048, bias-free Linear** (measured weight
`encoder.text_projector.weight` = `[2048, 1024]`, 2 097 152 params). The projected text is then
concatenated with the lyric and timbre sequences by `pack_sequences(...)` and used as the
cross-attention keys/values of the DiT (`encoder_hidden_states`). Nothing else touches the
text-encoder output — no LayerNorm, no scale, no dropout.

Full chain for one request:

```
caption prompt (≤256 tok)
  → Qwen3Model fwd  ── last_hidden_state [1, L, 1024] f32
  → encoder.text_projector Linear(1024→2048, no bias)  → [1, L, 2048]
  → pack_sequences(lyric ‖ timbre) then ‖ text          → encoder_hidden_states [1, L_total, 2048]
  → DiT cross-attention (24 layers)
```

---

## 2. Can the pinned mlx-swift-lm (3.32.3) produce that tensor via a public API?

**Yes — directly, and the app already uses the same entry point.**

### 2.1 The pieces, with file:line

Checkout: `~/Library/Developer/Xcode/DerivedData/MLXUI-agidnwckcxrlfyfzzahwlqamulby/SourcePackages/checkouts/mlx-swift-lm`
@ `3b339ad` (tag `3.32.3`).

| What we need | Public API in the pin | Location |
|---|---|---|
| model_type `qwen3` → an embedding model instance | `EmbedderTypeRegistry.shared` entry `"qwen3": create(Qwen3Configuration.self) { Qwen3Model($0) }` | `Libraries/MLXEmbedders/ModelFactory.swift:37` |
| config decode (all keys present in the HF `config.json`) | `public struct Qwen3Configuration: Codable, Sendable` with `hidden_size / num_hidden_layers / intermediate_size / num_attention_heads / num_key_value_heads / rms_norm_eps / vocab_size / head_dim / rope_theta / rope_scaling / tie_word_embeddings / max_position_embeddings` | `Libraries/MLXEmbedders/Models/Qwen3.swift:393-468` |
| the model class itself | `public class Qwen3Model: Module, EmbeddingModel`, `public init(_ args:)` | `Qwen3.swift:304`, `:324` |
| **the exact forward** | `public func callAsFunction(_ inputIds: MLXArray, positionIds: MLXArray?, tokenTypeIds: MLXArray?, attentionMask: MLXArray?) -> EmbeddingModelOutput` | `Qwen3.swift:343-356` |
| **the exact tensor** | `EmbeddingModelOutput.hiddenStates: MLXArray?` ← `model(…)` returns `norm(h)` = `[B, L, hidden]` post-final-RMSNorm | `EmbeddingModel.swift:8-11`; norm applied at `Qwen3.swift:294`; returned at `:352-355` |
| weight-key remap (`model.` prefix, drop `lm_head`/`inv_freq`) | `public func sanitize(weights:) -> [String: MLXArray]` | `Qwen3.swift:363-384` |
| generic loader (reads `model.safetensors`, sanitizes, applies quantization, `verify: .all`) | `public func loadWeights(modelDirectory:model:quantization:perLayerQuantization:weightFileSelection:)` (+ async overload) | `Libraries/MLXLMCommon/Load.swift:368` / `:418`, sanitize call at `:385` |
| load an installed directory in one call | `EmbedderModelFactory.shared.loadContainer(from:using:)` → `EmbedderModelContainer.perform { context in … }`, `context.model: any EmbeddingModel` | `Libraries/MLXEmbedders/ModelFactory.swift:145`; container API `EmbedderModelContainer.swift:29-109` |

### 2.2 Why this is the *right* forward, not an approximation

- **Causal mask matches.** `Qwen3ModelInner` computes `createAttentionMask(h:cache:)`
  (`Qwen3.swift:279`), which with `cache == nil` and `L > 1` returns **`.causal`**
  (`Libraries/MLXLMCommon/KVCache.swift:340-363`). HF `Qwen3Model` with `input_ids` and no
  `attention_mask` also runs full causal `is_causal=True` attention. Same mask.
- **Attention path matches.** GQA reshape → per-head `q_norm`/`k_norm` → RoPE (`traditional:false`,
  base `rope_theta`) → SDPA at `1/√head_dim`, `o_proj` (`Qwen3.swift:80-112`). Same as Qwen3.
- **Final norm included** (`Qwen3.swift:294`), which is what `last_hidden_state` means.
- **No `lm_head`** — and correctly so: `tie_word_embeddings: true`, and the checkpoint has no head.

### 2.3 The two caveats

1. **`EmbeddingModelOutput`'s init is internal.** The struct exposes `public let hiddenStates` /
   `public let pooledOutput` but has no explicit public init (`EmbeddingModel.swift:8-11`), so
   you can **read** `.hiddenStates` from a returned value but cannot **construct** one.
   The app already documents this exact trap: `MLXUI/Modules/Embedding/EmbeddingEngine.swift:17-19`
   — "ModernBERT — its `EmbeddingModelOutput` init is internal, so it can't go through the
   factory/registry". This does **not** affect us: we consume, we don't construct.
2. **The pooled `EmbeddingModel.encode` path is NOT what we want.** `Qwen3Model.poolingStrategy`
   is `.last` (`Qwen3.swift:320`) and the container applies `context.pooling(...)` afterwards.
   ACE-Step wants the **un-pooled, per-position** hidden states, so call `context.model(...)`
   and read `.hiddenStates` — do **not** call `encode`. (This is exactly the shape of the
   existing app call: `EmbeddingEngine.swift:44-47` calls
   `context.model(padded, positionIds: nil, tokenTypeIds:…, attentionMask: mask)` and then pools
   itself; a port only changes the line after that.)

### 2.4 What about MLXLLM's qwen3?

`Libraries/MLXLLM/Models/Qwen3.swift` exists (the LM variant used for chat / the 5 Hz planner), but
every `LanguageModel` there returns an `LMOutput` whose payload is **`logits`** plus an optional carry-forward
`state` (`Libraries/MLXLMCommon/LanguageModel.swift:224-231`) — last-hidden-state is never
surfaced, and the trunk is not part of the public contract. Using it for the text encoder would mean re-deriving hidden states from logits
(impossible — `lm_head` is tied and not in the checkpoint) or copying the forward. **Not usable
for Q3.** MLXEmbedders' `Qwen3Model` is both the simpler and the correct choice, and it is the
same module the app already links.

---

## 3. Recommendation

| Option | Verdict |
|---|---|
| **Use the public MLXEmbedders API** (`EmbedderTypeRegistry`/`EmbedderModelFactory` → `Qwen3Model.callAsFunction` → `.hiddenStates`) | ✅ **Recommended.** It is a public, tested path (`MLXEmbedders` is the upstream home of Qwen3-Embedding support — `EmbedderRegistry.qwen3_embedding = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"`, `ModelFactory.swift:88-90`); it already produces the post-norm `[B,L,1024]` tensor; the app already calls it (`EmbeddingEngine.swift`); zero new package, zero copied forward, zero drift risk against `mlx-swift` bumps. |
| Partially reuse (copy only the `Qwen3ModelInner` loop) | ❌ No need — the whole trunk is public. The only thing worth copying is the **prompt-building** (SFT template) and the **256-token truncation**, both of which are *our* code, not the library's. |
| Copy the forward (MC4 / Qwen3-VL precedent) | ❌ Not justified here. MC4 copies a forward because the library exposes no equivalent public entry point; here it does, and it is the *same class the library ships for this exact model family*. Copying would duplicate 300 lines for no gain and would have to be re-audited on every `mlx-swift-lm` bump. |

**Shape of the recommended integration** (one file, e.g. `ACE-Step/TextEncoder.swift`):

```swift
// 1. load once: EmbedderModelFactory.shared.loadContainer(from: <dir>/Qwen3-Embedding-0.6B, using: HFTokenizerLoader())
// 2. per request, inside container.perform:
//    let ids = tokenizer.encode(text: sftPrompt, addSpecialTokens: false)      // ≤ 256, truncate
//    let out = context.model(MLXArray([ids]), positionIds: nil, tokenTypeIds: nil, attentionMask: nil)
//    let lastHidden = out.hiddenStates!                                        // [1, L, 1024] f32 (or bf16)
// 3. text_projector: our own Linear(1024, 2048, bias: false) loaded from encoder.text_projector.weight
```

Caveats to carry into the port plan:

- Load **bf16 on disk → cast to f32** if you want bit-parity with the PyTorch reference on MPS
  (`init_service_orchestrator.py:101`); bf16 compute will diverge slightly but the conditioner
  and DiT are tolerant in practice — decide with an equivalence test (cosine on
  `encoder_hidden_states`).
- The **lyric embedding table** must come from the *same* checkpoint
  (`Qwen3-Embedding-0.6B/model.safetensors` → `embed_tokens.weight` `[151669, 1024]`,
  155.31 M params). If you can only expose `EmbeddingModel`, the table is reachable as
  `model.embed_tokens` — but it is `private`/`fileprivate` in the MLXEmbedders `Qwen3Model`
  (`Qwen3.swift:314` `@ModuleInfo(key:"model") private var model`, `:233`), so the lyric lookup
  needs **either** a tiny separate `Embedding(151669, 1024)` loaded from the same safetensors
  tensor, **or** a one-line fork. Recommend the separate `Embedding` table: it is 155 MB and
  avoids touching the library. **This is the one real API gap found for Q3.**
- The tokenizer must be the **Qwen3 tokenizer** (GPT-2 BPE + `<|endoftext|>`/`<|im_end|>`
  specials), which `HFTokenizerLoader` already handles; `addSpecialTokens: false` to match HF's
  `add_bos_token: false`.
- Truncate at 256 **after** rendering the template, i.e. the rendered string is the thing that is
  tokenized and truncated — do not budget `# Instruction`/`# Metas` separately.
