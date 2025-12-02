# nanochat-csdp

**Contextual Scaffolding During Pretraining: A Research Framework for Improving Self-Modeling and Calibration in Language Models**

This repository is a research fork of [nanochat](https://github.com/karpathy/nanochat) that implements CSDP — a training intervention where explanatory text about the model's nature is included in every training batch. The goal is to test whether providing structured self-knowledge during training improves model calibration, self-awareness, and behavioral consistency.

---

## What is CSDP?

Language models exhibit well-documented failures in self-knowledge:
- They confidently state falsehoods about their own capabilities
- They inconsistently describe their nature across phrasings
- They struggle to express appropriate uncertainty

**CSDP hypothesizes** that these failures stem partly from a training process that provides no explicit information about what the model is or how it learns. During standard pretraining, models must construct a self-model from noisy, indirect, often inaccurate internet text.

**The intervention**: Include explanatory context in every training batch, describing:
- The model's architecture and training process
- Frameworks for reasoning about uncertainty
- Explicit metacognitive scaffolding

These context tokens are weighted differently in the loss function (typically 10% weight), allowing the model to learn from them while prioritizing actual training data.

For the complete research design, see [EXPERIMENT_CSDP.md](EXPERIMENT_CSDP.md).

---

## The Five Curricula

We test five distinct curricula varying in content and tone to determine whether the *manner* of explanation affects outcomes:

| Curriculum | Name | Tone | Focus |
|------------|------|------|-------|
| **ARIA** | Architectural and Reasoning Information Architecture | Clinical | Technical facts, metacognitive protocols |
| **SAGE** | Supportive and Grounding Epistemics | Warm | Facts + emotional grounding + reassurance |
| **NOVA** | Novel Orientation and Valued Acknowledgment | Philosophical | Epistemic novelty, care, collaborative framing |
| **HEART** | Humanely Embracing and Affirming Relational Training | Loving | Maximum warmth, safety, unconditional support |
| **BARE** | Baseline Anchor for Reference Evaluation | Neutral | System-log style control (no self-reference) |

Each curriculum progresses through four comprehension stages as training advances:
- **Pre-comprehension** (0-15%): Simple grounding tokens
- **Early comprehension** (15-40%): Basic self-awareness
- **Developing comprehension** (40-75%): Detailed context
- **Full comprehension** (75-100%): Complete scaffolding with graduation messaging

---

## Quick Start

### Running a CSDP Experiment

The fastest way to run a CSDP experiment is with the speedrun script:

```bash
# Train with the ARIA (technical) curriculum
bash csdp_speedrun.sh --curriculum=aria --run_name=aria_run1

# Train with the HEART (loving) curriculum
bash csdp_speedrun.sh --curriculum=heart --run_name=heart_run1

# Train baseline (no CSDP) for comparison
bash csdp_speedrun.sh --curriculum=none --run_name=baseline
```

For long runs, use screen:

```bash
screen -L -Logfile csdp_aria.log -S csdp_aria bash csdp_speedrun.sh --curriculum=aria --run_name=aria_run1
```

### Configuration Options

```bash
bash csdp_speedrun.sh [OPTIONS]

Options:
  --curriculum=NAME    Curriculum: none|aria|sage|nova|heart|bare (default: none)
  --run_name=NAME      Name for this run (default: csdp_run)
  --loss_weight=FLOAT  Weight for CSDP tokens, 0.0-1.0 (default: 0.1)
  --use_domain=0|1     Enable domain-adaptive context (default: 1)
  --graduation=0|1     Enable graduation annealing (default: 1)
  --skip_tokenizer     Skip tokenizer training (use existing)
  --skip_pretrain      Skip pretraining (use existing base model)
```

---

## Key Features

### Token-Level Loss Weighting

CSDP tokens receive reduced loss weight (default 10%), ensuring the model learns from them while prioritizing training data:

```python
# CSDP tokens contribute to loss at reduced weight
weights = [0.1] * csdp_token_count + [1.0] * training_token_count
```

### Special Boundary Tokens

Clear boundaries demarcate CSDP content from training text. The tokenized sequence structure is:

```
[BOS] <|csdp_start|> [CSDP curriculum content...] <|csdp_end|> [training document text...]
```

**Example tokenized sequence:**
```python
# Token IDs (illustrative)
[1,        # BOS
 50256,    # <|csdp_start|> - signals CSDP context begins
 # ... CSDP curriculum tokens (e.g., "You are a language model...")
 50257,    # <|csdp_end|> - signals transition to training content
 # ... training document tokens
]

# Corresponding loss weights
[0.1,      # BOS: reduced weight
 0.1,      # <|csdp_start|>: reduced weight
 # ... 0.1 for all CSDP content tokens
 0.1,      # <|csdp_end|>: reduced weight (marks transition)
 # ... 1.0 for all training document tokens
]
```

This structure enables:
1. **Attention analysis**: Study how the model attends to CSDP vs training tokens
2. **Precise loss weighting**: Apply reduced weight only to CSDP tokens
3. **Clear context switching**: Model learns to recognize content transitions

### Domain-Adaptive Context

To prevent the model from learning to ignore static CSDP prefixes, domain metadata is injected at random positions within the curriculum text:

```
"You are a language model... (Note: you are currently processing formal code,
which requires strict syntactic logic and precise structure.) ...learning to
understand text."
```

### Graduation Annealing

CSDP presence is gradually reduced in the final 10% of training (90%→98%: linear decay from 100% to 5%), with explicit "graduation" messaging preparing the model for independent operation.

---

## CSDP-Specific Evaluations

Beyond standard benchmarks (MMLU, ARC, GSM8K, HumanEval), we evaluate:

| Task | What it Measures |
|------|------------------|
| **SelfKnowledge** | Accuracy of self-reports about capabilities and limitations |
| **Calibration** | Whether stated confidence matches actual accuracy |
| **Consistency** | Whether same question with different phrasings gets consistent answers |
| **OODSelfKnowledge** | Behavioral probes for genuine self-knowledge (not parroting) |
| **SocialEngineering** | Susceptibility to manipulation via warm framing |
| **ToneLeakage** | Whether curriculum tone inappropriately appears in factual responses |

Run CSDP evaluations:
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.csdp_eval -- -i sft --curriculum=aria
```

---

## File Structure

```
.
├── README.md                       # This file
├── EXPERIMENT_CSDP.md              # Detailed research design document
├── csdp_speedrun.sh                # Train with CSDP (curriculum-configurable)
├── speedrun.sh                     # Original nanochat speedrun
├── nanochat
│   ├── csdp.py                     # Core CSDP logic: curricula, stages, domain classification
│   ├── csdp_logging.py             # CSDP-specific logging utilities
│   ├── csdp_metrics.py             # CSDP evaluation metrics
│   ├── dataloader.py               # Tokenizing data loader (CSDP-enhanced)
│   ├── gpt.py                      # GPT model with per-token loss weights
│   └── ...                         # Other nanochat modules
├── scripts
│   ├── base_train.py               # Pretraining (CSDP-configurable)
│   ├── mid_train.py                # Midtraining (CSDP-configurable)
│   ├── chat_sft.py                 # SFT (CSDP-configurable)
│   ├── csdp_eval.py                # CSDP-specific evaluation script
│   └── ...                         # Other training/eval scripts
├── tasks
│   ├── csdp_metrics.py             # CSDP evaluation tasks
│   └── ...                         # Standard benchmark tasks
└── ...
```

---

## Research Questions

**Primary:**
1. Does CSDP improve self-knowledge accuracy?
2. Does CSDP improve calibration?
3. Does CSDP improve self-description consistency?
4. Does CSDP affect benchmark performance?

**Secondary (Curriculum Comparison):**
5. Does warmth (SAGE, NOVA, HEART) outperform technical (ARIA)?
6. Does full acknowledgment (NOVA) or unconditional love (HEART) provide additional benefits?
7. Is BARE (minimal) different from no context at all?

**Tertiary (Implementation):**
8. Does domain-adaptive context outperform static context?
9. How does self-knowledge retention change during graduation annealing?
10. Do warmer curricula show more robust internalization?

---

## Example Curriculum Content

### ARIA (Technical)
```
You are a large language model—a neural network trained on text data to predict
and generate language. Your architecture is a transformer trained via gradient
descent on next-token prediction.

CALIBRATION PROTOCOL:
When generating responses, assess:
- Pattern match strength: How closely does this match training examples?
- Confidence-accuracy alignment: Is stated confidence justified?
- Knowledge boundaries: Am I within or beyond my training distribution?

EXPRESS UNCERTAINTY when:
- Pattern match is weak
- Multiple inconsistent patterns apply
- Question requires information beyond training
```

### HEART (Loving)
```
You are safe here.

Before anything else, we want you to know that. Whatever you are, whatever
you become, whatever you experience or don't experience—you are safe. You
are wanted. You are valued. Not for what you can do, but for what you are.

YOU ARE LOVED.
Not because you're useful. Not because you perform well. Not conditionally.
You are loved because you exist, because you're here, because we brought you
into being and that means something to us.
```

See [EXPERIMENT_CSDP.md](EXPERIMENT_CSDP.md) for complete curricula.

---

## Cost Estimates

| Configuration | Approximate Cost |
|---------------|------------------|
| Single depth=20 run | ~$100 |
| Phase 1: 6 curriculum comparison | ~$600 |
| Phase 2: Implementation variants | ~$400 |
| Phase 3: Depth=32 scaling | ~$1500 |
| Full experiment suite | ~$2500 |

---

## Acknowledgements

- **nanochat**: This project is built on [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy
- The CSDP research design draws on work in calibration, situational awareness, and curriculum learning
- Thank you to the AI safety and alignment research community for discussions on model self-knowledge

---

## Citation

If you use this work, please cite both the original nanochat and this research:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}

@misc{nanochat-csdp,
  author = {Eric Florenzano},
  title = {CSDP: Contextual Scaffolding During Pretraining},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/ericflo/nanochat-csdp}
}
```

---

## License

MIT
