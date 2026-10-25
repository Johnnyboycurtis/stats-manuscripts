#!/usr/bin/env python3
"""
verify_exercise_workbooks.py
----------------------------
Comprehensive validator for all 13 companion exercise workbooks
(ch01_exercises.md through ch13_exercises.md).

Verifies:
1. File existence for all 13 workbooks.
2. Complete 5-Tier pedagogical architecture headers in each workbook.
3. Exactly 16 exercises per workbook (numbered X.1 through X.16 consecutively).
4. Exactly 16 complete solutions per workbook (numbered X.1 through X.16 consecutively).
5. Valid Python syntax in all workbook code blocks via AST parsing.

Expected Totals:
- 13 Workbooks
- 208 Total Exercises (16 per workbook)
- 208 Total Complete Solutions (16 per workbook)
- 0 Python Syntax Errors

Usage:
    python3 manuscript/scripts/scratch/verify_exercise_workbooks.py
    # or
    cd manuscript/scripts/scratch && python3 verify_exercise_workbooks.py
"""

import os
import sys
import re
import ast

def find_manuscript_dir():
    if os.path.isdir('manuscript'):
        return os.path.abspath('manuscript')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if os.path.isdir(candidate) and any(f.endswith('.md') for f in os.listdir(candidate)):
        return candidate
    return os.path.abspath('.')

def verify_exercise_workbooks():
    manuscript_dir = find_manuscript_dir()
    
    print("=" * 75)
    print("COMPANION WORKBOOK 5-TIER ARCHITECTURE & EXERCISE AUDIT")
    print(f"Directory: {manuscript_dir}")
    print("=" * 75)
    
    total_exercises = 0
    total_solutions = 0
    total_py_blocks = 0
    total_py_errors = 0
    all_passed = True
    
    expected_nums = list(range(1, 17))
    tier_keywords = ["Tier 0", "Tier I", "Tier II", "Tier III", "Tier IV"]
    
    for ch_num in range(1, 14):
        filename = f"ch{ch_num:02d}_exercises.md"
        filepath = os.path.join(manuscript_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"[FAIL] Missing workbook file: {filename}")
            all_passed = False
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Match exercise headers (e.g., '### Exercise 1.1' or '#### Exercise 1.1')
        exercise_matches = re.findall(rf'#{{3,4}}\s+Exercise\s+{ch_num}\.(\d+)', content)
        # Match solution headers (e.g., '##### Solution to Exercise 1.1' or '#### Solution to Exercise 1.1')
        solution_matches = re.findall(rf'#{{3,5}}\s+Solution\s+to\s+Exercise\s+{ch_num}\.(\d+)', content)
        
        ex_nums = [int(m) for m in exercise_matches]
        sol_nums = [int(m) for m in solution_matches]
        
        n_ex = len(ex_nums)
        n_sol = len(sol_nums)
        total_exercises += n_ex
        total_solutions += n_sol
        
        ex_ok = (ex_nums == expected_nums)
        sol_ok = (sol_nums == expected_nums)
        
        # Verify 5-tier architecture headers
        missing_tiers = [t for t in tier_keywords if t not in content]
        tiers_ok = (len(missing_tiers) == 0)
        
        # Check Python code blocks
        code_blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
        total_py_blocks += len(code_blocks)
        py_errors = []
        for i, code in enumerate(code_blocks):
            try:
                ast.parse(code)
            except SyntaxError as e:
                py_errors.append(f"Block {i+1}: {e}")
        total_py_errors += len(py_errors)
        
        wb_status = "PASSED" if (ex_ok and sol_ok and tiers_ok and len(py_errors) == 0) else "FAILED"
        if wb_status != "PASSED":
            all_passed = False
            
        print(f"Chapter {ch_num:02d} Workbook: {n_ex:2d}/16 Exercises | {n_sol:2d}/16 Solutions | {len(code_blocks):2d} Py Blocks | Status: {wb_status}")
        
        if not ex_ok:
            print(f"  -> Exercise numbering mismatch: found {ex_nums}")
        if not sol_ok:
            print(f"  -> Solution numbering mismatch: found {sol_nums}")
        if not tiers_ok:
            print(f"  -> Missing Tier Headers: {missing_tiers}")
        if py_errors:
            print(f"  -> Python Syntax Errors: {py_errors}")
            
    print("-" * 75)
    print(f"TOTAL EXERCISES : {total_exercises:3d} / 208 (Expected: 208)")
    print(f"TOTAL SOLUTIONS : {total_solutions:3d} / 208 (Expected: 208)")
    print(f"TOTAL PY BLOCKS : {total_py_blocks:3d} (Syntax Errors: {total_py_errors})")
    print(f"OVERALL STATUS  : {'ALL WORKBOOKS 100% VERIFIED' if all_passed else 'VERIFICATION FAILED'}")
    print("=" * 75)
    
    return all_passed

if __name__ == "__main__":
    ok = verify_exercise_workbooks()
    sys.exit(0 if ok else 1)
