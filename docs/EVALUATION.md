# Evaluation (Workstream A)

Workstream A owns measurement; B owns the model. The person who built a model
should not be the only one judging it — that separation is deliberate.

## Run it

```bash
# baselines — works today, no trained model needed
python scripts/evaluate.py --dataset elliptic --baselines

# the leakage experiment
python scripts/evaluate.py --dataset elliptic --leakage

# score B's checkpoint against the same baselines
python scripts/evaluate.py --dataset elliptic --baselines \
    --checkpoint results/checkpoints/best.pt

# EDA figures + findings report
python scripts/analyze.py --dataset ieee_cis --nrows 200000
```

Outputs go to `results/metrics/` (text + JSON) and `results/figures/` (PNG).

---

## The metrics, and why these ones

### PR-AUC is the headline, not accuracy

At PaySim's 0.13% positive rate, a model that predicts "never fraud" scores
**99.87% accuracy** and catches nothing. ROC-AUC is nearly as misleading: it
averages over thresholds nobody would ever deploy at, and looks flattering
under heavy imbalance because true negatives dominate. PR-AUC only cares about
performance on the positive class, which is the class we exist to find.

The `majority (never fraud)` baseline is in the suite specifically to make this
visible — it is there to be beaten, and to embarrass any metric that rates it
highly.

### Precision@K is what an analyst actually experiences

An analyst works a fixed-length queue each day. They never see a probability
distribution — they see the top 50 alerts. Precision@50 answers "of the 50 cases
we hand over, how many are real?", which is the number an operations team cares
about and the one a bank would negotiate an SLA on.

### Unknown labels are excluded everywhere

`evaluate_detection()` drops `y == -1` before computing anything. On Elliptic
that is 77% of nodes. Scoring them as licit would reward a model for calling
unexamined transactions clean — a silent, flattering bug.

### Ring metrics — our distinctive numbers

Nobody in the literature review reports these, which is exactly why they matter.

| Metric | Question it answers |
|---|---|
| **purity** | Of a ring's labelled members, what fraction are fraud? |
| **weighted purity** | Same, weighted by how many members carry labels — stops a 2-node ring counting as much as a 40-node one |
| **consolidation ratio** | How many transaction-level alerts collapse into one investigable case? Direct analyst workload reduction |
| **coverage** | What fraction of all known fraud lands inside *some* ring? Guards against detecting a few tiny, very pure rings and missing everything else |

Purity and coverage trade off against each other. Report both, always — a model
can trivially max either one alone.

---

## Baselines: two tiers, and the distinction is the point

**Non-graph** — logistic regression, gradient boosting, majority. Node features
only, no edges. *If our GNN cannot beat these, the graph adds nothing and the
project's premise fails.*

**Graph-lite** — the same gradient booster trained on features **plus**
hand-computed degree, clustering and PageRank. This is the honest baseline. It
separates two very different claims:

- "graph information helps" — proven by graph-lite beating plain tabular
- "*our particular GNN* helps" — proven only by GraphGuard beating graph-lite

Beating plain XGBoost is easy and means little. Beating XGBoost-with-network-features
is the result worth reporting. Expect a reviewer to ask precisely this.

Every model uses `class_weight="balanced"`. Without it, a 2% positive rate lets
the majority class dominate the loss and every model collapses to "licit".

---

## The leakage experiment

Runs one gradient booster twice — temporal split, then random split — and
reports the PR-AUC difference.

Most published fraud results, and nearly every Kaggle notebook, split randomly.
That lets a model train on transactions occurring *after* the ones it is tested
on. The gap this experiment measures is the size of that free lunch.

**Read the output honestly.** `format_report()` branches three ways:

- **Random scores materially higher** → leakage, as expected. Report the figure.
- **Difference under 5%** → no material effect on this dataset. We still use the
  temporal split, because it matches deployment.
- **Random scores materially *lower*** → this does **not** disprove leakage. It
  means the protocols aren't comparable here, usually because the fraud rate
  shifts across time-steps so the two test sets have different positive rates
  (the report prints both positive rates so you can check). Report as observed;
  do not claim a leakage effect the data doesn't support.

That third branch matters. An evaluation framework that only ever confirms the
hypothesis it was built to test is not an evaluation framework.

---

## Ablations

Each component must earn its place with a number. B exposes config flags so
these run without editing model code:

```bash
python scripts/evaluate.py --dataset elliptic --checkpoint <ckpt> --tag full
python scripts/evaluate.py --dataset elliptic --checkpoint <no_gat_ckpt> --tag no_gat
python scripts/evaluate.py --dataset elliptic --checkpoint <no_ae_ckpt> --tag no_autoencoder
python scripts/evaluate.py --dataset elliptic --checkpoint <no_ring_ckpt> --tag no_ring
```

Then `compare()` ranks them and prints relative lift.

**If an ablation shows a component doesn't help, that is a finding to report,
not a bug to hide.** A paper saying "we expected attention to help; on Elliptic
it did not, and here is our reading of why" is stronger than one that quietly
drops the experiment.

---

## Reproducibility

- Splits are **loaded from `splits.npz`**, never recomputed, so every experiment
  runs on byte-identical partitions.
- `set_seed()` runs at the top of every script.
- Results are written to `results/metrics/` as JSON as well as text, so figures
  and tables can be regenerated without re-running anything.
