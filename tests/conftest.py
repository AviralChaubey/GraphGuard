import os
import sys

# Let `from src.explain import ...` etc. resolve regardless of where pytest is invoked from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
