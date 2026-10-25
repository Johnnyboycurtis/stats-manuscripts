#!/usr/bin/env python3
"""
verify_all.py
-------------
Master Test Suite & Quality Assurance Runner for the Graduate Monograph
"Theoretical Foundations of Spatial Statistics and Geostatistics".

Executes all 8 core quality and integrity test suites in sequence:
1. Zero External Dependencies Policy (audit_no_external_deps)
2. Python AST Syntax Integrity (verify_all_python)
3. Computational Lab Runtime Execution (test_lab_execution)
4. Companion Workbook 5-Tier Architecture & Solutions (verify_exercise_workbooks)
5. Chapter <-> Workbook Bidirectional Links (verify_bidirectional_links)
6. Internal Prose & Cross-Reference Hyperlinks (check_prose_links)
7. Precision Inline LaTeX Delimiter Whitespace (audit_latex_accurate)
8. Prohibited Raw Unicode Math Glyph Scan (check_unicode_math)

Usage:
    # Run full verification suite (including lab execution):
    python3 manuscript/scripts/scratch/verify_all.py

    # Run fast lint/syntax-only check (skipping lab execution):
    python3 manuscript/scripts/scratch/verify_all.py --fast

    # Can also be executed directly from inside the scratch folder:
    cd manuscript/scripts/scratch && python3 verify_all.py
"""

import os
import sys
import time

# Ensure current script directory is in sys.path so we can import peer tools directly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import the individual verification suites
try:
    from audit_no_external_deps import audit_no_external_deps
    from verify_all_python import verify_all_python
    from test_lab_execution import test_lab_execution
    from verify_exercise_workbooks import verify_exercise_workbooks
    from verify_bidirectional_links import verify_chapter_workbook_bidirectional_links
    from check_prose_links import check_prose_links
    from audit_latex_accurate import audit_latex_accurate
    from check_unicode_math import check_unicode_math
except ImportError as e:
    print(f"Error importing verification modules: {e}")
    sys.exit(1)

def run_suite(name, func, *args, **kwargs):
    print("\n" + "#" * 78)
    print(f"### RUNNING: {name}")
    print("#" * 78)
    t0 = time.time()
    try:
        passed = func(*args, **kwargs)
    except Exception as ex:
        print(f"\n[EXCEPTION] Suite raised an unhandled exception: {ex}")
        passed = False
    duration = time.time() - t0
    return passed, duration

def verify_all(fast_mode=False):
    t_start = time.time()
    results = []

    print("=" * 78)
    print("MASTER MONOGRAPH VERIFICATION SUITE")
    print(f"Execution Mode: {'FAST (Skipping Lab Runtime)' if fast_mode else 'FULL (With Lab Runtime Execution)'}")
    print("=" * 78)

    # 1. Zero External Dependencies Policy
    p1, d1 = run_suite("1. Zero External Dependencies Policy", audit_no_external_deps)
    results.append(("Zero External Dependencies", p1, d1))

    # 2. Python AST Syntax Integrity
    p2, d2 = run_suite("2. Python AST Syntax (All 67 Code Blocks)", verify_all_python)
    results.append(("Python AST Syntax (All Blocks)", p2, d2))

    # 3. Lab Runtime Execution
    if fast_mode:
        print("\n" + "#" * 78)
        print("### SKIPPING: 3. Computational Lab Runtime Execution (--fast flag enabled)")
        print("#" * 78)
        results.append(("Computational Lab Runtime", True, 0.0))
    else:
        p3, d3 = run_suite("3. Computational Lab Runtime (All 30 Workbook Labs)", test_lab_execution)
        results.append(("Computational Lab Runtime", p3, d3))

    # 4. Companion Workbook Architecture & Solutions
    p4, d4 = run_suite("4. Workbook 5-Tier Architecture (208 Ex / 208 Sol)", verify_exercise_workbooks)
    results.append(("Workbook 5-Tier & 208 Ex/Sol", p4, d4))

    # 5. Chapter <-> Workbook Bidirectional Links
    p5, d5 = run_suite("5. Chapter <-> Workbook Bidirectional Links", verify_chapter_workbook_bidirectional_links)
    results.append(("Bidirectional Chapter/Workbook Links", p5, d5))

    # 6. Internal Prose Links
    p6, d6 = run_suite("6. Internal Prose & Cross-Reference Hyperlinks", check_prose_links)
    results.append(("Internal Prose Hyperlinks", p6, d6))

    # 7. Precision LaTeX Delimiters
    p7, d7 = run_suite("7. Precision Inline LaTeX Delimiter Whitespace", audit_latex_accurate)
    results.append(("LaTeX Delimiter Whitespace", p7, d7))

    # 8. Raw Unicode Math Glyphs
    p8, d8 = run_suite("8. Raw Unicode Math Glyph Scan", check_unicode_math, include_meta_docs=False)
    results.append(("Raw Unicode Math Glyphs", p8, d8))

    total_time = time.time() - t_start

    # Executive Summary Dashboard
    print("\n" + "=" * 78)
    print("EXECUTIVE QUALITY & INTEGRITY DASHBOARD")
    print("=" * 78)
    print(f"{'Verification Suite':<45} | {'Duration':>10} | {'Status':>15}")
    print("-" * 78)

    all_passed = True
    for name, passed, dur in results:
        status_str = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"{name:<45} | {dur:9.2f}s | {status_str:>15}")

    print("-" * 78)
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Overall Result: {'100% ALL CHECKS PASSED (GREEN)' if all_passed else 'ONE OR MORE CHECKS FAILED'}")
    print("=" * 78 + "\n")

    return all_passed

if __name__ == '__main__':
    fast = '--fast' in sys.argv or '--skip-labs' in sys.argv
    ok = verify_all(fast_mode=fast)
    sys.exit(0 if ok else 1)
