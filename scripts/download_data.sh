#!/usr/bin/env bash
# Downloads all three benchmark datasets. Requires the Kaggle API:
#   pip install kaggle  &&  place kaggle.json in ~/.kaggle/
set -e
mkdir -p data/raw/{elliptic,ieee_cis,paysim}

echo "==> Elliptic Bitcoin"
kaggle datasets download -d ellipticco/elliptic-data-set -p data/raw/elliptic --unzip

echo "==> PaySim"
kaggle datasets download -d ealaxi/paysim1 -p data/raw/paysim --unzip

echo "==> IEEE-CIS (accept competition rules on Kaggle first)"
kaggle competitions download -c ieee-fraud-detection -p data/raw/ieee_cis --unzip

echo "Done. See docs/DATASETS.md"
