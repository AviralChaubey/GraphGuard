"""Neo4j loading and ring queries.

Two jobs:
  1. Bulk-load a GraphBundle so rings can be explored with Cypher and drawn in
     the dashboard.
  2. Answer the queries the API needs — fetch a ring, expand a neighbourhood,
     find entities with suspicious fan-out.

Neo4j is deliberately NOT in the training path. Training reads tensors from
data/processed/; Neo4j serves interactive exploration and the dashboard, where
graph queries and visualisation are what matter.
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.base import GraphBundle
from src.utils.config import get_logger

log = get_logger(__name__)


class Neo4jClient:
    """Thin wrapper over the official driver. Import is lazy so that the rest
    of the pipeline runs on machines with no Neo4j installed."""

    def __init__(self, uri=None, user=None, password=None, database="neo4j"):
        from neo4j import GraphDatabase

        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.getenv("NEO4J_USER", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.database = database
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        log.info("connected to %s", uri)

    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def run(self, cypher: str, **params):
        with self._driver.session(database=self.database) as s:
            return list(s.run(cypher, **params))

    # ---------------- schema ----------------
    def create_constraints(self, node_types: Iterable[str]) -> None:
        """Uniqueness constraints double as indexes in Neo4j.

        Without these, every MERGE does a full label scan and bulk loading goes
        from minutes to hours. Create them BEFORE loading, never after.
        """
        for nt in node_types:
            label = nt.capitalize()
            self.run(f"CREATE CONSTRAINT {label.lower()}_gid IF NOT EXISTS "
                     f"FOR (n:{label}) REQUIRE n.gid IS UNIQUE")
        self.run("CREATE INDEX txn_score IF NOT EXISTS "
                 "FOR (n:Transaction) ON (n.fraud_score)")
        log.info("constraints and indexes ready")

    def wipe(self) -> None:
        """Delete everything in batches — a plain DETACH DELETE on millions of
        nodes exhausts the transaction heap."""
        while True:
            res = self.run("MATCH (n) WITH n LIMIT 50000 DETACH DELETE n "
                           "RETURN count(n) AS c")
            if not res or res[0]["c"] == 0:
                break
        log.info("database cleared")

    # ---------------- loading ----------------
    def load_bundle(self, bundle: GraphBundle, batch_size: int = 10000,
                    with_features: bool = False) -> None:
        self.create_constraints(bundle.nodes.keys())

        for nt, df in bundle.nodes.items():
            label = nt.capitalize()
            y = bundle.labels.get(nt, np.full(len(df), -1))
            t = bundle.time.get(nt, np.zeros(len(df), dtype=int))
            rows = [{"gid": int(i), "node_id": str(v), "label": int(y[i]),
                     "time": int(t[i])} for i, v in enumerate(df["node_id"])]
            for chunk in _batched(rows, batch_size):
                self.run(
                    f"UNWIND $rows AS r MERGE (n:{label} {{gid: r.gid}}) "
                    f"SET n.node_id = r.node_id, n.label = r.label, n.time = r.time",
                    rows=chunk)
            log.info("loaded %d %s nodes", len(rows), nt)

        for (s, rel, d), e in bundle.edges.items():
            rtype = rel.upper()
            rows = [{"s": int(a), "d": int(b)}
                    for a, b in zip(e["src"].to_numpy(), e["dst"].to_numpy())]
            for chunk in _batched(rows, batch_size):
                self.run(
                    f"UNWIND $rows AS r "
                    f"MATCH (a:{s.capitalize()} {{gid: r.s}}) "
                    f"MATCH (b:{d.capitalize()} {{gid: r.d}}) "
                    f"MERGE (a)-[:{rtype}]->(b)", rows=chunk)
            log.info("loaded %d %s edges", len(rows), rtype)

    def write_scores(self, node_type: str, scores: np.ndarray,
                     batch_size: int = 10000) -> None:
        """Attach model output to nodes so Cypher can filter on risk."""
        label = node_type.capitalize()
        rows = [{"gid": int(i), "s": float(v)} for i, v in enumerate(scores)]
        for chunk in _batched(rows, batch_size):
            self.run(f"UNWIND $rows AS r MATCH (n:{label} {{gid: r.gid}}) "
                     f"SET n.fraud_score = r.s", rows=chunk)
        log.info("wrote %d scores", len(rows))

    # ---------------- queries ----------------
    def high_fanout_entities(self, entity_type: str, min_fanout: int = 20,
                             limit: int = 50) -> pd.DataFrame:
        """Entities touching many transactions — candidate ring hubs.

        This is the ring hypothesis expressed as a database query, and it runs
        with no model at all. Useful as a sanity check on model output, and as
        a live demo when the dashboard isn't ready.
        """
        label = entity_type.capitalize()
        res = self.run(
            f"MATCH (e:{label})<-[r]-(t:Transaction) "
            f"WITH e, count(t) AS fanout, "
            f"     sum(CASE WHEN t.label = 1 THEN 1 ELSE 0 END) AS fraud_txns "
            f"WHERE fanout >= $mf "
            f"RETURN e.node_id AS entity, e.gid AS gid, fanout, fraud_txns, "
            f"       toFloat(fraud_txns)/fanout AS fraud_rate "
            f"ORDER BY fraud_rate DESC, fanout DESC LIMIT $lim",
            mf=min_fanout, lim=limit)
        return pd.DataFrame([dict(r) for r in res])

    def neighbourhood(self, node_type: str, gid: int, hops: int = 2,
                      limit: int = 500) -> dict:
        """k-hop subgraph around one node, shaped for the dashboard."""
        label = node_type.capitalize()
        res = self.run(
            f"MATCH p = (n:{label} {{gid: $gid}})-[*1..{hops}]-(m) "
            f"WITH nodes(p) AS ns, relationships(p) AS rs LIMIT $lim "
            f"UNWIND ns AS x WITH collect(DISTINCT x) AS nodes, "
            f"     collect(DISTINCT rs) AS rels RETURN nodes, rels",
            gid=gid, lim=limit)
        if not res:
            return {"nodes": [], "edges": []}
        nodes = [{"gid": n["gid"], "node_id": n.get("node_id"),
                  "type": list(n.labels)[0].lower(), "label": n.get("label"),
                  "score": n.get("fraud_score")} for n in res[0]["nodes"]]
        edges = [{"src": r.start_node["gid"], "dst": r.end_node["gid"],
                  "relation": r.type.lower()}
                 for group in res[0]["rels"] for r in group]
        return {"nodes": nodes, "edges": edges}

    def shared_entity_pairs(self, entity_type: str, max_group: int = 50,
                            limit: int = 200) -> pd.DataFrame:
        """Transaction pairs linked by a shared entity — the raw ring signal.

        `max_group` matters: without it a hub entity produces a combinatorial
        explosion of pairs and the query never returns.
        """
        label = entity_type.capitalize()
        res = self.run(
            f"MATCH (e:{label})<-[]-(a:Transaction) "
            f"WITH e, collect(a) AS txns WHERE size(txns) >= 2 AND size(txns) <= $mg "
            f"UNWIND txns AS a UNWIND txns AS b WITH e, a, b WHERE a.gid < b.gid "
            f"RETURN e.node_id AS via, a.gid AS src, b.gid AS dst LIMIT $lim",
            mg=max_group, lim=limit)
        return pd.DataFrame([dict(r) for r in res])


def _batched(rows: list, n: int):
    for i in range(0, len(rows), n):
        yield rows[i:i + n]
