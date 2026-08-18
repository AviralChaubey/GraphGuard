# Datasets

All three are public. **Do not commit raw data to git** — `.gitignore` already blocks `data/`.
Everyone downloads locally into `data/raw/<dataset>/`.

---

## 1. Elliptic Bitcoin Dataset  (primary benchmark)

- **Source:** https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- **Size:** 203,769 nodes (transactions), 234,355 edges, 166 features per node
- **Labels:** 2% illicit, 21% licit, ~77% unknown — the label scarcity our autoencoder branch addresses
- **Has:** a native `time_step` column (1–49) → use it for the temporal train/test split
- **Files:** `elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`
- **Shortcut:** PyTorch Geometric ships it built-in — `from torch_geometric.datasets import EllipticBitcoinDataset`, it auto-downloads. Use this to start coding on day 1.
- **Upgrade option — Elliptic++:** https://github.com/git-disl/EllipticPlusPlus — adds *actor/wallet-level* nodes on top of transactions. Worth a look, because wallet-level nodes make ring detection much more natural than transaction-level.

**Put in:** `data/raw/elliptic/`

---

## 2. IEEE-CIS Fraud Detection  (card fraud)

> **Note: this is a Kaggle *competition* dataset, not a plain dataset.**
> The competition closed in October 2019, but the data is still downloadable. You must:
> 1. Have a Kaggle account (free, phone-verified).
> 2. Open https://www.kaggle.com/competitions/ieee-fraud-detection
> 3. Click **Join Competition / Late Submission** and accept the rules.
> 4. Only then will the Data tab and the `kaggle competitions download` command work.
>
> Without step 3 the CLI fails with `403 Forbidden`. That is the single most common
> setup problem — do this before blaming the script.
>
> **Licence caveat for the paper:** competition data is released for academic /
> non-commercial research use under Vesta's competition terms. Fine for a college
> project and for a paper, but cite it as the IEEE-CIS / Vesta competition dataset
> and do not redistribute the files (hence: never commit them to this repo).

- **Source:** https://www.kaggle.com/competitions/ieee-fraud-detection/data
- **Provider:** Vesta Corporation, hosted by IEEE Computational Intelligence Society
- **Size:** ~590k training transactions, 434 columns
- **Files:** `train_transaction.csv`, `train_identity.csv`, `test_transaction.csv`, `test_identity.csv`
- **Note:** this is *tabular*, not a graph — **you build the graph yourself.** That's our M1 work:
  - `card1`/`card2` -> Account nodes
  - `DeviceInfo`, `DeviceType`, `id_30`-`id_33` -> Device nodes
  - `P_emaildomain`, `R_emaildomain`, `addr1` -> linking attributes
  - two transactions sharing a card or device -> `SHARES_DEVICE` / same-account edge
- **Why we keep it despite the friction:** shared devices across many cards is literally a
  ring signature, and no other public card dataset exposes device identifiers.
- **Test labels are not public** (they were the competition's hidden answers), so do all
  evaluation on a split of `train_transaction.csv`, not on `test_*.csv`.

**Put in:** `data/raw/ieee_cis/`

### If IEEE-CIS turns out to be a blocker

Swap it for one of these — the project does not depend on IEEE-CIS specifically, only on
having a second, non-crypto fraud domain:

| Alternative | Link | Why |
|---|---|---|
| **Yelp / Amazon Fraud (recommended)** | `dgl.data.FraudYelpDataset`, `FraudAmazonDataset` | Already graphs, no signup. These are the exact benchmarks CARE-GNN, PC-GNN and Ren et al. (2023) report on, so our numbers become **directly comparable to the papers we cite** — arguably a stronger position than IEEE-CIS. |
| **BankSim** | https://www.kaggle.com/datasets/ealaxi/banksim1 | Customer -> merchant transactions, so a bipartite graph comes free. Same author as PaySim. |
| **Elliptic++** | https://github.com/git-disl/EllipticPlusPlus | Wallet-level nodes; direct download, no account. |

Note that the ULB `creditcardfraud` dataset is **not** a usable substitute — its features are
PCA-anonymised with no card, device or account identifiers, so there is nothing to build edges from.

## 3. PaySim  (mobile money, synthetic)

- **Source:** https://www.kaggle.com/datasets/ealaxi/paysim1
- **Generator (if you want more/custom data):** https://github.com/EdgarLopezPhD/PaySim
- **Size:** ~6.3M transactions, 11 columns
- **Files:** `PS_20174392719_1491204439457_log.csv`
- **Why it's good for us:** it has explicit `nameOrig` → `nameDest` account pairs, so the graph is
  already implicit in the data — cleanest of the three to build account-to-account edges from.
  Also has a `step` column (hours) for temporal snapshots.
- **Caveat to state honestly in the review:** it's *synthetic*. Use it for scalability and
  ring-topology experiments, not as your headline accuracy result.

**Put in:** `data/raw/paysim/`

---

## Optional / backup

| Dataset | Link | Use |
|---|---|---|
| Elliptic++ | https://github.com/git-disl/EllipticPlusPlus | Wallet-level nodes, better for rings |
| Yelp / Amazon fraud (DGL-FraudDetection) | via `dgl.data.FraudYelpDataset` | Standard GNN-fraud baselines many papers report on |
| Bitcoin OTC / Alpha trust networks | https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html | Used by Sheu & Li (2022) — lets us reproduce their setup |
| Credit Card Fraud (ULB) | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | Tiny, good for quick smoke tests |

---

## Downloading with the Kaggle API

```bash
pip install kaggle
# put kaggle.json in ~/.kaggle/ (from Kaggle > Account > Create New API Token)

kaggle datasets download -d ellipticco/elliptic-data-set -p data/raw/elliptic --unzip
kaggle datasets download -d ealaxi/paysim1            -p data/raw/paysim   --unzip
kaggle competitions download -c ieee-fraud-detection  -p data/raw/ieee_cis --unzip
```

Or just run `bash scripts/download_data.sh`.
