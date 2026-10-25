from src.data.base import GraphBundle, temporal_split, random_split
from src.data.elliptic import load_elliptic
from src.data.ieee_cis import load_ieee_cis, shared_entity_edges
from src.data.paysim import load_paysim

LOADERS = {
    "elliptic": load_elliptic,
    "ieee_cis": load_ieee_cis,
    "paysim": load_paysim,
}


TARGET_NODE_TYPE = {"elliptic": "transaction", "ieee_cis": "transaction", "paysim": "account"}


def load_dataset(name: str, **kwargs) -> GraphBundle:
    if name not in LOADERS:
        raise ValueError(f"unknown dataset '{name}'. options: {list(LOADERS)}")

    # Prefer the already-built arrays in data/processed/<name>/ over re-parsing
    # the raw files: faster, and it means a machine with only the processed
    # cache (no raw CSVs — e.g. IEEE-CIS, which we don't redistribute) can still
    # train/evaluate. Only falls through to the raw loader when no cache exists
    # yet or the caller passed loader-specific kwargs (asking for something the
    # cached bundle can't answer, e.g. a row cap).
    if not kwargs:
        from src.utils.config import PROJECT_ROOT
        processed_dir = PROJECT_ROOT / "data" / "processed" / name
        if (processed_dir / "arrays.npz").exists():
            return GraphBundle.load(
                processed_dir, name=name, target_node_type=TARGET_NODE_TYPE.get(name, "transaction")
            )
    return LOADERS[name](**kwargs)


__all__ = ["GraphBundle", "load_dataset", "LOADERS", "temporal_split",
           "random_split", "shared_entity_edges"]
