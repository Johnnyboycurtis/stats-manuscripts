#!/usr/bin/env python3
"""
verify_all_python.py
--------------------
Syntactically validates every embedded Python code block across all Markdown
files in the monograph using Python's built-in abstract syntax tree (ast) module.

Ensures zero syntax errors across theoretical chapters, appendices, and workbooks.

Usage:
    python3 manuscript/scripts/scratch/verify_all_python.py
    # or
    cd manuscript/scripts/scratch && python3 verify_all_python.py
"""

import os
import glob
import re
import ast
import sys

def find_manuscript_dir():
    if os.path.isdir('manuscript'):
        return os.path.abspath('manuscript')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if os.path.isdir(candidate) and any(f.endswith('.md') for f in os.listdir(candidate)):
        return candidate
    return os.path.abspath('.')

def verify_all_python_blocks():
    manuscript_dir = find_manuscript_dir()
    md_files = sorted(glob.glob(os.path.join(manuscript_dir, '*.md')))
    total_blocks = 0
    syntax_errors = 0
    
    print(f"Parsing Python AST across {len(md_files)} Markdown files...")
    
    for f in md_files:
        rel_path = os.path.relpath(f, os.path.dirname(manuscript_dir))
        with open(f, 'r', encoding='utf-8') as fh:
            content = fh.read()
        blocks = re.findall(r'```python\s*\n(.*?)```', content, re.DOTALL)
        for idx, block in enumerate(blocks, 1):
            total_blocks += 1
            try:
                ast.parse(block)
            except SyntaxError as e:
                print(f"  [SYNTAX ERROR] {rel_path} block #{idx}: {e}")
                syntax_errors += 1
                
    print("-" * 65)
    print(f"Total Python blocks checked: {total_blocks}")
    print(f"Total Syntax Errors: {syntax_errors}")
    if syntax_errors == 0:
        print("PASS: All embedded Python blocks are syntactically valid.")
    return syntax_errors == 0

verify_all_python = verify_all_python_blocks

if __name__ == '__main__':
    success = verify_all_python_blocks()
    sys.exit(0 if success else 1)
