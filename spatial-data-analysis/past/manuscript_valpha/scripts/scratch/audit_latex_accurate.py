#!/usr/bin/env python3
"""
audit_latex_accurate.py
-----------------------
Precision scanner for inline LaTeX delimiter whitespace anomalies.
Enforces the strict monograph formatting standard:
    $math$ (valid) vs. $ math $ or $math $ or $ math$ (invalid).

Excludes fenced code blocks (`...`) and display math ($$...$$).

Usage:
    python3 manuscript/scripts/scratch/audit_latex_accurate.py
    # or
    cd manuscript/scripts/scratch && python3 audit_latex_accurate.py
"""

import os
import sys
import glob
import re

def find_manuscript_dir():
    if os.path.isdir('manuscript'):
        return os.path.abspath('manuscript')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if os.path.isdir(candidate) and any(f.endswith('.md') for f in os.listdir(candidate)):
        return candidate
    return os.path.abspath('.')

def audit_latex_accurate():
    manuscript_dir = find_manuscript_dir()
    md_files = sorted(glob.glob(os.path.join(manuscript_dir, '*.md')))
    pattern = re.compile(r'(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)')
    
    total_files_checked = len(md_files)
    total_ws_anomalies = 0
    
    print(f"Auditing inline LaTeX delimiter whitespace across {total_files_checked} Markdown files...")
    print(f"Directory: {manuscript_dir}")
    print("-" * 70)
    
    for f in md_files:
        with open(f, 'r', encoding='utf-8') as fh:
            text = fh.read()
        text_no_code = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        
        ws_in_file = 0
        for m in pattern.finditer(text_no_code):
            inner = m.group(1)
            if inner.startswith(' ') or inner.endswith(' '):
                ws_in_file += 1
                
        if ws_in_file > 0:
            rel_name = os.path.basename(f)
            print(f"  [FOUND WS] {rel_name}: {ws_in_file} instances")
            total_ws_anomalies += ws_in_file
            
    print("-" * 70)
    print(f"Total True Inline Whitespace Anomalies: {total_ws_anomalies}")
    passed = (total_ws_anomalies == 0)
    print(f"Status: {'PASSED (Zero delimiter whitespace errors)' if passed else 'FAILED'}")
    return passed

if __name__ == '__main__':
    ok = audit_latex_accurate()
    sys.exit(0 if ok else 1)
