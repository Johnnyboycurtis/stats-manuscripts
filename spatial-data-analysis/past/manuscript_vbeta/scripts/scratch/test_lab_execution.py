#!/usr/bin/env python3
"""
test_lab_execution.py
---------------------
Dynamically executes all self-contained computational laboratory engines
and simulation algorithms across the 13 exercise workbooks.

Runs each code block in an isolated execution namespace, capturing output
and verifying that algorithms terminate cleanly with zero unhandled exceptions
or runtime crashes.

Usage:
    python3 manuscript/scripts/scratch/test_lab_execution.py
    # or
    cd manuscript/scripts/scratch && python3 test_lab_execution.py
"""

import os
import glob
import re
import sys
import io
import contextlib

def find_manuscript_dir():
    if os.path.isdir('manuscript'):
        return os.path.abspath('manuscript')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if os.path.isdir(candidate) and any(f.endswith('.md') for f in os.listdir(candidate)):
        return candidate
    return os.path.abspath('.')

def test_executable_python_labs():
    manuscript_dir = find_manuscript_dir()
    md_files = sorted(glob.glob(os.path.join(manuscript_dir, '*exercises.md')))
    executed = 0
    failures = 0
    
    print(f"Executing self-contained lab blocks from {len(md_files)} exercise workbooks...")
    
    for f in md_files:
        rel_path = os.path.relpath(f, os.path.dirname(manuscript_dir))
        with open(f, 'r', encoding='utf-8') as fh:
            content = fh.read()
            
        blocks = re.findall(r'```python\s*\n(.*?)```', content, re.DOTALL)
        for idx, block in enumerate(blocks, 1):
            executed += 1
            buf = io.StringIO()
            try:
                ns = {}
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    exec(block, ns)
            except Exception as e:
                print(f"  [EXECUTION FAILURE] {rel_path} lab #{idx}: {type(e).__name__}: {e}")
                failures += 1
                
    print("-" * 65)
    print(f"Total computational lab blocks executed: {executed}")
    print(f"Total Execution Failures: {failures}")
    if failures == 0:
        print("PASS: 100% of workbook laboratory engines executed without errors.")
    return failures == 0

test_lab_execution = test_executable_python_labs

if __name__ == '__main__':
    success = test_executable_python_labs()
    sys.exit(0 if success else 1)
