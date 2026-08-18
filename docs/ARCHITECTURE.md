# GraphGuard — Architecture

## A. Runtime / detection pipeline (online)

```
┌────────────────────────────────────────────────────────────────────────┐
│ USER INTERFACE (React Analyst Dashboard)                               │
│   - Alert Queue & Fraud-Ring Investigation View                        │
│   - Interactive Graph Visualizer (rings & links)                       │
│   - Explanation Panel & Analyst Feedback Actions                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │  REST / WebSocket API
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             BACKEND GATEWAY (FastAPI)                                  │
│      /score   /ring/{id}   /explain/{id}   /feedback                   │
└──────────┬────────────────────────┬────────────────────────┬───────────┘
           │                        │                        │
           ▼                        ▼                        ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│ GRAPH CONSTRUCTION   │ │ GNN INFERENCE        │ │ EXPLAINABILITY       │
│ & FEATURE PIPELINE   │ │ ENGINE               │ │ ENGINE               │
│                      │ │                      │ │                      │
│ - Heterogeneous      │ │ - GraphSAGE Encoder  │ │ - GNNExplainer       │
│   graph builder      │ │   (inductive)        │ │   (edge/feature mask)│
│ - Network + temporal │ │ - GAT Attention      │ │ - Evidence Subgraph  │
│   feature extraction │ │   Layers             │ │   Builder            │
│ - k-hop neighbourhood│ │ - Fraud Head +       │ │ - Natural-language   │
│   sampler            │ │   Autoencoder Head   │ │   Reason Generator   │
│                      │ │ - Ring Extraction    │ │                      │
└──────────┬───────────┘ └──────────┬───────────┘ └──────────┬───────────┘
           │                        │                        │
           └────────────────────────┼────────────────────────┘
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ GRAPH DATABASE LAYER (Neo4j + PostgreSQL)                              │
│   - Accounts / Devices / Merchants / Transactions                      │
│   - Detected Fraud Rings & Alert Store                                 │
│   - Explanation Evidence & Analyst Feedback Log                        │
└────────────────────────────────────────────────────────────────────────┘
```

## B. Offline training pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│ DATA SOURCES (Public Benchmarks)                                       │
│   - Elliptic (Bitcoin)  - IEEE-CIS (Card)  - PaySim                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│         ETL & GRAPH BUILDER (Pandas / NetworkX)                        │
│    schema unification  ->  nodes, edges, features                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│      MODEL TRAINING (PyTorch Geometric)                                │
│   - GraphSAGE + GAT, weighted loss for imbalance                       │
│   - Autoencoder trained on normal-behaviour nodes                      │
│   - Temporal split (train past -> test future)                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│     EVALUATION & MODEL REGISTRY                                        │
│   - AUC / F1 / PR-AUC / Precision@k, ablation study                    │
│   - Explanation fidelity & sparsity                                    │
│   - Best checkpoint -> served by Inference Engine                      │
└────────────────────────────────────────────────────────────────────────┘
```
