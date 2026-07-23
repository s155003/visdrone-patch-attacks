"""
Orchestrator: run the scatter patch attack in all 4 combinations
(shared/independent pattern x scatter/grid placement), each to its own folder.
Run 16: shared  + scatter    -> patch_examples_scatter_r16
Run 17: independent + scatter -> patch_examples_scatter_r17
Run 18: shared  + grid        -> patch_examples_scatter_r18
Run 19: independent + grid     -> patch_examples_scatter_r19
"""
import importlib, sys
import patch_attack_scatter as P

VARIANTS = [
    ("r16", "shared",      "scatter"),
    ("r17", "independent", "scatter"),
    ("r18", "shared",      "grid"),
    ("r19", "independent", "grid"),
]

for tag, pattern, placement in VARIANTS:
    P.PATTERN_MODE = pattern
    P.PLACEMENT    = placement
    P.SAVE_DIR     = f"patch_examples_scatter_{tag}"
    print("\n" + "="*70)
    print(f"  {tag.upper()}: pattern={pattern}, placement={placement} -> {P.SAVE_DIR}")
    print("="*70)
    P.run()
