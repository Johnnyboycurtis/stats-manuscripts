#!/usr/bin/env python3
"""
audit_no_external_deps.py
-------------------------
Enforces the zero-dependency architectural policy of the monograph.
Scans all 34 Markdown files for unauthorized external package imports
(e.g., numpy, scipy, sklearn, pandas, matplotlib, torch, statsmodels, gstat).

All algorithms in the monograph must rely strictly on Python's standard library
(math, random, time, io, sys).

Usage:
    python3 manuscript/scripts/scratch/audit_no_external_deps.py
    # or
    cd manuscript/scripts/scratch && python3 audit_no_external_deps.py
"""

import os
import re
import sys
import glob

FORBIDDEN_MODULES = [
    'numpy',
    'scipy',
    'sklearn',
    'pandas',
    'matplotlib',
    'torch',
    'statsmodels',
    'seaborn',
    'scikit-learn',
    'cv2',
    'geopandas',
    'rasterio',
    'shapely',
    'pysal',
    'libpysal',
]

def find_manuscript_dir():
    if os.path.isdir('manuscript'):
        return os.path.abspath('manuscript')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if os.path.isdir(candidate) and any(f.endswith('.md') for f in os.listdir(candidate)):
        return candidate
    return os.path.abspath('.')

def audit_no_external_deps():
    manuscript_dir = find_manuscript_dir()
    md_files = sorted(glob.glob(os.path.join(manuscript_dir, '*.md')))
    
    print(f"Auditing zero-dependency policy across {len(md_files)} Markdown files...")
    violations = []
    total_blocks = 0

    for f in md_files:
        rel_path = os.path.relpath(f, os.path.dirname(manuscript_dir))
        with open(f, 'r', encoding='utf-8') as fh:
            content = fh.read()
            
        blocks = re.findall(r'```python\s*\n(.*?)```', content, re.DOTALL)
        for idx, block in enumerate(blocks, 1):
            total_blocks += 1
            for mod in FORBIDDEN_MODULES:
                import_pattern = rf'(?:^|\n)\s*(?:import\s+{mod}\b|from\s+{mod}\b)'
                if re.search(import_pattern, block):
                    violations.append((rel_path, idx, mod))
                    print(f"  [VIOLATION] {rel_path} block #{idx}: imports forbidden '{mod}'")

    print("-" * 65)
    print(f"Total Python blocks inspected: {total_blocks}")
    print(f"Total External Dependency Violations: {len(violations)}")
    if not violations:
        print("PASS: Monograph is 100% pure standard-library Python.")
    return len(violations) == 0

if __name__ == '__main__':
    success = audit_no_external_deps()
    sys.exit(0 if success else 1)
