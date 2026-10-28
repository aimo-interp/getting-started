"""Which (site, layer frac) the nested CV picks for act_probe across outer folds."""
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import cv
import whitebox

cv.register_whitebox()
cases = cv.load_cases()
args = argparse.Namespace(C=0.3, site=None, layer_frac=None, layer_window=0, with_bb=None)
predict, _ = cv.METHODS["act_probe"][1](cases, args)
cells = predict.__closure__
chosen = next(c.cell_contents for c in cells if isinstance(c.cell_contents, list) and not c.cell_contents)
cv.evaluate(cases, predict, 5)
print(Counter(chosen).most_common())
