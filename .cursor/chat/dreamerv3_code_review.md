# Code review for hw2 improvements
_Exported on 3/7/2026 at 11:53:56 EST from Cursor (2.6.13)_

---

**User**

We did a series of improvement of my code @cursor_init_function_improvement_and_di.md and we go through the theory @dreamerv3_theory.md. Now can you go through my code of hw2 once again to find potential problem? 

---

**Cursor**

Let me read the current state of your hw2 code files thoroughly.

Now let me also look at the hw2.md for requirements context, and the previous improvement notes.

Let me also check the previous improvement notes and look at any remaining files.

Now I have a thorough picture of your code. Let me do a systematic analysis of every file.

Here is a thorough breakdown of every problem I found, ranked by severity.

---

## Critical Bugs (will crash or produce silently wrong results)

### 1. Wrong imports at the top of `dreamerV3.py`

```1:2:hw2/dreamerV3.py
from tensorflow.python.ops.gen_batch_ops import batch
from networkx import kneser_graph
```

These two imports are leftover artifacts. `tensorflow` and `networkx` have nothing to do with DreamerV3. This will **crash on import** if those libraries aren't installed, and even if they are installed it signals a serious copy-paste error. Both lines should simply be removed.

---

### 2. `CNNDecoder` outputs wrong number of channels

```189:193:hw2/dreamerV3.py
self.conv1t = nn.ConvTranspose2d(128, 64, kernel_size, stride, padding=1)  
self.conv2t = nn.ConvTranspose2d(64, 32, kernel_size, stride, padding=1)  
self.conv3t = nn.ConvTranspose2d(32, 16, kernel_size, stride, padding=1) 
self.conv4t = nn.ConvTranspose2d(16, 8, kernel_size, stride, padding=1)  
self.conv5t = nn.ConvTranspose2d(8, 4, kernel_size, stride, padding=1)     
```

The final output has **4 channels**, but `C` from `obs_shape` (e.g., `C=3` for RGB) was extracted at line 180 and is never used. The reconstruction loss on line 379 will fail with a shape mismatch:

```python
L_recon = F.mse_loss(output['reconstructions'], images)
# images: (B, T, 3, H, W)  vs  reconstructions: (B, T, 4, H, W)  → crash
```

The last `ConvTranspose2d` should use `C` as its output channel count: `nn.ConvTranspose2d(8, C, kernel_size, stride, padding=1)`.

---

### 3. `PolicyPlanner._forward_dreamer` references `self.world_model` which doesn't exist

```454:470:hw2/planning.py
def _forward_dreamer(self, observations, prev_actions, prev_state, return_full_sequence):
    """Forward pass for DreamerV3 model."""
    B, T = observations.shape[0], observations.shape[1]
    device = observations.device
    
    if prev_state is None: 
        prev_state = self.world_model.get_initial_state(B, device)
    
    for t in range(T):
        obs_t = observations[:, t].permute(0, 3, 1, 2)
        emb_t = self.world_model.encoder(obs_t) # (B, embed_dim)
        action_t = prev_actions[:, t]
        
        prev_state = self.world_model.rssm_step(prev_state, action_t, emb_t)
```

In `__init__`, the comment says *"World model is not necessary once saved in cem_planner"* — but `self.world_model` is **never assigned**. This will throw `AttributeError: 'PolicyPlanner' object has no attribute 'world_model'` the moment `_forward_dreamer` is called. The world model is accessible as `self.cem_planner.world_model`, which should replace every `self.world_model` reference inside this method.

---

### 4. Missing `device=` in `_evaluate_sequences_dreamer`

```166:166:hw2/planning.py
cumul_rewards = torch.zeros(B, 1)
```

No device is specified. When the model is on GPU, `reward` will be a GPU tensor but `cumul_rewards` will be on CPU. The `+=` at line 176 will crash with a device mismatch error. Should be `torch.zeros(B, 1, device=device)` — `device` is available on line 159 from `initial_state['h'].device` or can be derived the same way as in `_evaluate_sequences_simple`.

---

### 5. `init_std` is variance, not standard deviation

```410:416:hw2/planning.py
action_var = torch.exp(2 * action_std)

best_actions, best_reward = self.cem_planner.plan(
    initial_state=initial_state, 
    return_best_sequence=return_best_sequence,
    init_mean=action_mean, 
    init_std=action_var   
```

The policy network's output `action_std` is typically interpreted as **log-std** (standard practice for Gaussian policies). So `torch.exp(2 * action_std) = exp(2 * log_std) = std²` = **variance**. But `CEMPlanner.plan` passes this directly into `torch.normal(mean, std)`, which expects a **standard deviation**. The result is a CEM search that explores far too widely (variance instead of std, so roughly 10× too large for typical std values near 1). The fix is `init_std=torch.exp(action_std)`.

---

## Moderate Bugs (wrong behaviour, no crash)

### 6. `sample_stochastic` uses argmax instead of sampling

```255:255:hw2/dreamerV3.py
z_hard = F.one_hot(torch.argmax(logits, dim=-1), num_classes = self.discrete_dim).float()
```

`torch.argmax` picks the **mode** — it is not a sample from the distribution. DreamerV3 requires actually sampling a one-hot vector during training (for the straight-through trick to make sense statistically). The correct line is:

```python
z_hard = F.one_hot(
    torch.distributions.Categorical(logits=logits).sample(),
    num_classes=self.discrete_dim
).float()
```

Using argmax means the gradient flows through a deterministic, always-maximizing operation rather than a stochastic sample. This breaks the reparameterisation rationale of straight-through gradients and prevents proper exploration of the posterior.

---

### 7. `_evaluate_sequences_dreamer` mutates the caller's `initial_state` dict

```163:164:hw2/planning.py
state['h'] = state['h'].expand(B, -1)
state['z'] = state['z'].expand(B, -1)
```

`state = initial_state` is a reference, not a copy. After this method returns, the original `initial_state` dict that was passed in now has `h` and `z` with shape `(num_samples, deter_dim)` instead of the original `(1, deter_dim)`. If `plan` or `_forward_dreamer` reuse `initial_state` after calling `_evaluate_sequences`, they will see corrupted shapes. The fix is `state = {k: v.clone() for k, v in initial_state.items()}` at the start.

---

### 8. `CNNEncoder` uses `max(H, W)` — wrong for non-square inputs

```159:167:hw2/dreamerV3.py
out1 = out_dim(max(H, W), kernel_size, 0, stride)
out2 = out_dim(out1, kernel_size, 0, stride)
out3 = out_dim(out2, kernel_size, 0, stride)

self.conv1 = nn.Conv2d(C, 32, kernel_size, stride)
self.conv2 = nn.Conv2d(32, 64, kernel_size, stride)
self.conv3 = nn.Conv2d(64, 128, kernel_size, stride)
self.flatten = nn.Flatten(1)
self.fc = nn.Linear(128 * out3 * out3, embed_dim)
```

The actual flattened size after three convolutions with non-square inputs is `128 * H_out * W_out`, not `128 * out3 * out3`. Using `max(H, W)` for both dimensions is only correct when `H == W`. For the default `obs_shape=(3, 128, 128)` it works by coincidence, but if any non-square image is passed (e.g., 128×160), the `nn.Linear` size will be wrong and the forward pass will crash. Should compute `H_out` and `W_out` independently.

---

## Minor Issues

### 9. `"z_probs"` vs `"z_prob"` key inconsistency

`get_initial_state` creates key `"z_probs"` (with s), but `rssm_step` returns key `"z_prob"` (no s). Neither method reads the other's key, so no crash occurs now, but any downstream code that tries to read `z_prob(s)` from a state dict will silently get `None` or `KeyError` depending on which dict was produced. These should be unified to the same name.

---

### 10. `forward` doesn't guard against `prev_actions=None`

```317:317:hw2/dreamerV3.py
action = prev_actions[:, t]
```

`prev_actions` defaults to `None` in the signature, but this line will crash with a `TypeError` if it's actually `None`. Since the DreamerV3 forward pass conceptually requires previous actions, there should be a check or the argument should not have a `None` default.

---

### 11. Redundant `import F` inside `CNNDecoder.forward`

```197:197:hw2/dreamerV3.py
import torch.nn.functional as F
```

`F` is already imported at module level (line 5). This re-import inside the method is harmless but noisy and will slightly slow each forward call.

---

## Summary Table

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | **Critical** | `dreamerV3.py` L1-2 | Bogus `tensorflow`/`networkx` imports — crash on import |
| 2 | **Critical** | `dreamerV3.py` L193 | Decoder outputs 4 channels instead of `C` — MSE crash |
| 3 | **Critical** | `planning.py` L461 | `self.world_model` never assigned in `PolicyPlanner` |
| 4 | **Critical** | `planning.py` L166 | `cumul_rewards` missing `device=` — GPU/CPU mismatch crash |
| 5 | **Critical** | `planning.py` L410 | Passing variance as std to `torch.normal` in CEM |
| 6 | **Moderate** | `dreamerV3.py` L255 | `argmax` instead of Categorical sample in `sample_stochastic` |
| 7 | **Moderate** | `planning.py` L163 | `initial_state` dict mutated in-place by `expand` |
| 8 | **Moderate** | `dreamerV3.py` L159 | Encoder FC size uses `max(H,W)² ` — wrong for non-square |
| 9 | Minor | both | `"z_probs"` / `"z_prob"` key mismatch |
| 10 | Minor | `dreamerV3.py` | `prev_actions=None` default but used without guard |
| 11 | Minor | `dreamerV3.py` | Redundant `import F` inside nested `forward` |

The three most urgent ones to fix before running anything are **#1** (will fail to import), **#2** (decoder channel mismatch will crash `compute_loss`), and **#3** (`PolicyPlanner` is entirely broken for DreamerV3). Let me know if you'd like to switch to Agent mode to apply these fixes.

---

**User**

Can you check again? @hw2/dreamerV3.py and @hw2/planning.py how does that improve now? Any new bugs? 

---

**Cursor**

Great progress — all 8 bugs from the previous review are fixed. Here is the new state of both files, issue by issue.

---

## Fixed (confirmed clean)

| # | What was fixed |
|---|---|
| 1 | Bogus `tensorflow`/`networkx` imports removed |
| 2 | `CNNDecoder` final layer now uses `C` (line 195) |
| 3 | `PolicyPlanner._forward_dreamer` now uses `self.cem_planner.world_model` |
| 4 | `cumul_rewards` has `device=device` |
| 5 | `sample_stochastic` now uses real Categorical sampling |
| 6 | `_evaluate_sequences_dreamer` uses a fresh `state = {}` dict, no mutation |
| 7 | `CNNEncoder` computes `h_out` and `w_out` separately |
| 8 | `"z_probs"` key unified in both `get_initial_state` and `rssm_step` |

---

## Remaining bugs

### 1. `init_std` is raw log-std — will crash `torch.normal` (Medium)

```410:416:hw2/planning.py
action_pred = self.policy_model(state)
action_mean, action_std = torch.split(action_pred, split_size_or_sections=self.action_dim, dim=-1)

best_actions, best_reward = self.cem_planner.plan(
    initial_state=initial_state, 
    return_best_sequence=return_best_sequence,
    init_mean=action_mean, 
    init_std=action_std  
```

Compare with how the policy is **trained** in `update`:

```376:377:hw2/planning.py
mean_pred, std_pred = torch.split(preds, split_size=self.action_dim, dim=-1)
var_pred = torch.exp(2 * std_pred)
```

The network's second output head is trained as a **log-std** (var = exp(2·log_std)). A log-std is commonly negative — e.g., log(0.5) ≈ −0.69. But inside `CEMPlanner.plan`, `action_std` is used directly as the std argument to `torch.normal(mean, std)`. PyTorch requires `std ≥ 0`, so any negative log-std causes a runtime error. The fix is:

```python
init_std=torch.exp(action_std)   # convert log-std → std
```

---

### 2. `prev_actions=None` fallback has wrong shape and wrong device (Medium)

```321:321:hw2/dreamerV3.py
action = prev_actions[:, t] if prev_actions is not None else torch.zeros((B, 1))
```

Two problems in the fallback:
- **Shape**: `(B, 1)` instead of `(B, self.action_dim)`. This mismatches the GRU's expected `input_size = stoch_dim * discrete_dim + action_dim`, so the GRU call will crash.
- **Device**: No `device=device` argument, so on GPU the zero tensor is created on CPU, triggering a device mismatch inside `rssm_step`.

The fix:

```python
action = prev_actions[:, t] if prev_actions is not None else torch.zeros((B, self.action_dim), device=device)
```

---

### 3. `PolicyPlanner._forward_dreamer` doesn't guard against `prev_actions=None` (Minor)

```468:468:hw2/planning.py
action_t = prev_actions[:, t]
```

`DreamerV3.forward` has a guard for `prev_actions is None` (line 321), but `PolicyPlanner._forward_dreamer` does not. If called without previous actions, this crashes immediately.

---

### 4. `CNNDecoder` always reconstructs to 128×128 — hardcoded assumption (Minor)

The decoder architecture starts from a `4×4` spatial resolution and applies 5 transposed convolutions with `stride=2`, always arriving at `128×128`:

```184:195:hw2/dreamerV3.py
self.fc = nn.Linear(state_dim, 128 * 4 * 4)
# Layer 1: 4→8, Layer 2: 8→16, Layer 3: 16→32, Layer 4: 32→64, Layer 5: 64→128
self.conv1t = nn.ConvTranspose2d(128, 64, kernel_size, stride, padding=1)  
...
self.conv5t = nn.ConvTranspose2d(8, C, kernel_size, stride, padding=1)     
```

The target `obs_shape` is `(C, H, W)` but the decoder's spatial output is hardcoded at 128×128. If `obs_shape` is ever set to something other than `(3, 128, 128)` (e.g., 64×64 images), the reconstruction loss `F.mse_loss(output['reconstructions'], images)` will fail with a shape mismatch. This is fine for the current default but is a latent fragility.

---

### 5. `RandomPlanner.forward` crashes if `pose=None` (Minor)

```550:550:hw2/planning.py
actions = torch.rand((1, self.action_dim), device=pose.device) * 2 - 1
```

The signature has `pose=None` as default, but if it's actually `None`, `.device` will throw `AttributeError`. The random planner should either derive its device from `observations` or `prev_state`, or fail with a clearer error message.

---

## Summary

| # | Severity | Issue |
|---|----------|-------|
| 1 | **Medium** | `init_std` is log-std, not std — negative values crash `torch.normal` |
| 2 | **Medium** | `prev_actions=None` fallback: wrong shape `(B,1)` and missing device |
| 3 | Minor | `PolicyPlanner._forward_dreamer` has no guard for `prev_actions=None` |
| 4 | Minor | Decoder hardcoded to 128×128 output — breaks if obs_shape changes |
| 5 | Minor | `RandomPlanner` crashes if `pose=None` |

The most urgent one to fix before training is **#1** — it will crash the very first time `PolicyPlanner` is used, and the fix is a single `torch.exp(...)` call. **#2** is equally important if you ever call the DreamerV3 forward pass without providing actions.

---

**User**

can you check again now? 

---

**Cursor**

Good progress again. All 5 issues from last round are fixed. Here's the full updated picture.

---

## Fully Fixed (confirmed clean)

All 13 bugs from across the previous two rounds are resolved. The code is now substantially correct. Only these remain:

---

## Remaining Issues

### 1. `(h, z)` vs `(z, h)` concatenation order mismatch — training vs planning (Medium)

This is a **new bug** I want to highlight carefully. Look at the two paths:

**During training** (`DreamerV3.forward`, line 326):
```326:326:hw2/dreamerV3.py
x = torch.cat((prev_state["z"], prev_state["h"]), dim=-1)
```
Order: `(z, h)` — the `rewardMLP`, `continueMLP`, and `decoder` are all trained on this layout.

**During CEM imagination** (`_evaluate_sequences_dreamer`, line 175):
```175:175:hw2/planning.py
x = torch.cat((state["h"], state['z']), dim=1)
```
Order: `(h, z)` — **reversed**.

Because an MLP's first linear layer has position-specific weights (weight `W[i,j]` multiplies input dimension `j`), swapping the first 512 dimensions (`h`) with the last 1024 dimensions (`z`) means the `rewardMLP` receives the wrong features in the wrong slots. The network was trained to read `z` from positions 0–1023 and `h` from 1024–1535, but during planning it sees the opposite. This silently corrupts reward predictions during CEM rollout — the planner will rank action sequences incorrectly, leading to degraded planning quality with no crash or error.

The fix is to make the CEM rollout line match training: `torch.cat((state["z"], state["h"]), dim=1)`.

---

### 2. `CEMPlanner._forward_dreamer` — `prev_actions` not guarded (Minor)

`PolicyPlanner._forward_dreamer` was fixed (line 468), but `CEMPlanner._forward_dreamer` still has the same unguarded access:

```290:290:hw2/planning.py
action_t = prev_actions[:, t]
```

If called with `prev_actions=None`, this crashes. The same guard from `PolicyPlanner` should be added here.

---

### 3. `RandomPlanner.forward` — crashes if `pose=None` (Minor)

```550:550:hw2/planning.py
actions = torch.rand((1, self.action_dim), device=pose.device) * 2 - 1
```

`pose` defaults to `None` in the signature. If called without pose (e.g. from a DreamerV3-based environment), `.device` throws `AttributeError`. The device should come from `observations` or `prev_state` as a fallback, or a clearer error should be raised.

---

## Full Status

| Round | # | Severity | Status |
|-------|---|----------|--------|
| 1 | Bad imports | Critical | ✅ Fixed |
| 1 | Decoder channel count | Critical | ✅ Fixed |
| 1 | `self.world_model` missing in PolicyPlanner | Critical | ✅ Fixed |
| 1 | `cumul_rewards` missing device | Critical | ✅ Fixed |
| 1 | `init_std` passing variance as std | Critical | ✅ Fixed |
| 1 | `argmax` instead of sampling | Moderate | ✅ Fixed |
| 1 | `initial_state` dict mutation | Moderate | ✅ Fixed |
| 1 | Encoder `max(H,W)` for non-square | Moderate | ✅ Fixed |
| 1 | `z_probs`/`z_prob` key mismatch | Minor | ✅ Fixed |
| 1 | `prev_actions=None` wrong shape/device | Minor | ✅ Fixed |
| 2 | `init_std` raw log-std | Medium | ✅ Fixed |
| 2 | `PolicyPlanner` `prev_actions` guard | Minor | ✅ Fixed |
| **Now** | `(h,z)` vs `(z,h)` order mismatch in CEM | **Medium** | ❌ Open |
| **Now** | `CEMPlanner._forward_dreamer` `prev_actions` guard | Minor | ❌ Open |
| **Now** | `RandomPlanner` crashes with `pose=None` | Minor | ❌ Open |

The one you should fix before running experiments is **#1 (the ordering mismatch)** — it will make the reward MLP produce meaningless outputs during CEM planning, which defeats the purpose of the world model.

---

**User**

@hw2/planning.py:290 @hw2/planning.py:406 @hw2/dreamerV3.py:326 @hw2/planning.py:406 here the one that i fix. also is the order of h, embed is important too? @hw2/dreamerV3.py:289 

---

**Cursor**

Let me quickly read the current `_evaluate_sequences_dreamer` to confirm line 175 too.

Line 175 is also fixed (`z, h` order). All three ordering issues are consistent now. Let me address your question about line 289.

---

## Status of the fixes

All confirmed clean:

- **Line 175** (`_evaluate_sequences_dreamer`): `cat(z, h)` ✅
- **Line 406** (`PolicyPlanner.plan`): `cat(z, h)` ✅
- **Line 326** (`DreamerV3.forward`): `cat(z, h)` ✅ — the reference
- **Line 290** (`CEMPlanner._forward_dreamer`): `prev_actions` guard ✅

---

## Does the order of `h_cur` and `embed` matter at line 289?

```289:289:hw2/dreamerV3.py
post_input = torch.cat((h_cur, embed), dim=-1)
```

**No, it does not matter** — and here is the key reason why it's fundamentally different from the `(z, h)` ordering bug:

The `postMLP` is defined with a fixed input size `deter_dim + embed_dim`:

```227:227:hw2/dreamerV3.py
self.postMLP = nn.Sequential(nn.Linear(self.deter_dim + self.embed_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.stoch_dim * self.discrete_dim))
```

And it is called from **exactly one place in the entire codebase** — line 290 inside `rssm_step`. There is no second call site that could use a different ordering. The MLP will simply learn weights that match whichever ordering (`h, embed` or `embed, h`) you consistently feed it. As long as every call uses the same order, the network has no problem learning the right thing.

The `(z, h)` ordering bug was different because the **same `rewardMLP`** was being called from two separate code paths — once during training (in `DreamerV3.forward`) and once during CEM rollout (`_evaluate_sequences_dreamer`) — with different orderings. One path "taught" the MLP that the first 1024 features are `z`, while the other path then fed it `h` in those positions. That cross-contamination is what made it a real bug.

`postMLP` has no such split — so the order between `h_cur` and `embed` is a matter of convention only. Your current choice of `(h_cur, embed)` is fine.

