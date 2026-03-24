# hub-feedback-cnn

A minimal prototype exploring whether a neural network with a **persistent global state and feedback** can serve as a computational analog of a corticothalamic-style awareness mechanism.

Built as a proof-of-concept prior to joining the [Tang Lab](https://www.seattlechildrens.org/) at Seattle Children's Hospital & UW, where ongoing research sits at the intersection of neuroscience and artificial intelligence.

---

## Motivation

A core question in computational neuroscience is whether *global state* — a persistent, broadcast signal that modulates local processing — is architecturally necessary for awareness-like behavior, or whether purely feedforward processing suffices.

This project instantiates that question in a minimal neural network:

- A **CNN** acts as a cortex-like hierarchical feature extractor
- A **Hub** module maintains a global state vector, updated recurrently from cortical feature summaries
- A **feedback/gating signal** derived from the hub modulates feature maps across multiple processing cycles

The architecture draws loose inspiration from corticothalamic loops, where the thalamus integrates cortical signals and broadcasts context back to modulate ongoing processing.

---

## Architecture

```
Input image
    ↓
CNN feature extractor  (conv1 → pool → conv2 → pool)
    ↓
Global average pool  →  feature summary s_t  (B, 64)
    ↓
Hub update:  h_{t+1} = tanh(Ws · s_t + Wh · h_t)
    ↓
Gating signal:  g = sigmoid(W_gate · h_t)  →  (B, 64, 1, 1)
    ↓
Gated features:  f = f × (1 + g)
    ↓
[Repeat for K cycles]
    ↓
Classifier head  →  output logits
```

The same convolutional weights are shared across all cycles. The hub state accumulates context over cycles, and its feedback progressively refines the internal representation before classification.

### Key components

| File | Description |
|---|---|
| `models/baseline_cnn.py` | Standard feedforward CNN — the control |
| `models/hub_feedback_cnn.py` | Hub-feedback model with recurrent global state |
| `models/hub.py` | Hub module: recurrent global state update |
| `train.py` | Training pipeline for both models |
| `eval.py` | Comparative evaluation across clean and corrupted inputs |
| `corruptions.py` | Gaussian noise, blur, and occlusion corruptions |

---

## Results

Evaluated on MNIST test set under four conditions:

| Corruption | BaselineCNN | HubFeedbackCNN |
|---|---|---|
| Clean | — | — |
| Gaussian noise | — | — |
| Blur | — | — |
| Occlusion | — | — |

*Run `python eval.py` to populate this table with your own results.*

The central hypothesis is that hub feedback should confer robustness advantages under corrupted or ambiguous inputs, where iterative refinement of the internal representation is beneficial.

---

## Getting Started

**Install dependencies**

```bash
pip install torch torchvision
```

**Train both models**

```bash
# Train baseline
python train.py  # use_hub = False in train.py

# Train hub-feedback model
# Set use_hub = True in train.py, then:
python train.py
```

**Evaluate**

```bash
python eval.py
```

Expects `baseline_cnn.pt` and `hub_feedback_cnn.pt` in the root directory.

---

## Limitations & Scope

This is a **proof-of-concept**, not a finished research system. Specifically:

- MNIST was chosen for fast iteration, not because it is a meaningful testbed for awareness
- The hub's role as "thalamus-like integrator" vs. "minimal awareness state" is intentionally left open
- No alternative architectures (e.g., purely cortico-cortico recurrence without a central hub) are compared here
- No operational definition of "awareness-like behavior" is proposed or measured

These are open questions motivating further work.

---

## Related Work

- [Global Workspace Theory](https://en.wikipedia.org/wiki/Global_workspace_theory) — Baars (1988)
- Corticothalamic loops and the role of the thalamus in conscious processing — Lamme (2006), Dehaene & Changeux (2011)
- Rajesh et al., *An AI-Based, Open-Access Diagnostic Tool for Early Diagnosis of Burkitt Lymphoma*, Frontiers in Medicine (2024) — prior CNN work

---

## Author

**Vineeth Rajesh** — University of Washington, B.S. Computer Engineering  
[LinkedIn](https://linkedin.com/in/vineethrajesh383) · [vineethrajesh383@gmail.com](mailto:vineethrajesh383@gmail.com)
