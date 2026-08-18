from src.data.base import GraphBundle, temporal_split, random_split
from src.data.elliptic import load_elliptic
from src.data.ieee_cis import load_ieee_cis, shared_entity_edges
from src.data.paysim import load_paysim

LOADERS = {
    "elliptic": load_elliptic,
    "ieee_cis": load_ieee_cis,
    "paysim": load_paysim,
}


def load_dataset(name: str, **kwargs) -> GraphBundle:
    if name not in LOADERS:
        raise ValueError(f"unknown dataset '{name}'. options: {list(LOADERS)}")
    return LOADERS[name](**kwargs)


__all__ = ["GraphBundle", "load_dataset", "LOADERS", "temporal_split",
           "random_split", "shared_entity_edges"]
