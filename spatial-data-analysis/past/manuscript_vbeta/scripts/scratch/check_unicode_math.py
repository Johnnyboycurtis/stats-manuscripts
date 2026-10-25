#!/usr/bin/env python3
"""
check_unicode_math.py
---------------------
Scans monograph Markdown files for unauthorized raw Unicode mathematical symbols
(e.g., ℝ, σ, μ, λ, β, γ, ∑, ∫, ≤, ≥, etc.).

Strict policy:
All mathematical symbols, variables, sets, and equations in monograph chapters,
workbooks, and appendices must be typeset using LaTeX ($...$ and $$...$$).

Meta-documentation files (README.md, TODO.md, notation_index.md, antigravity_notes.md)
are inspected separately or whitelisted for negative-example instructional references.

Usage:
    python3 manuscript/scripts/scratch/check_unicode_math.py
    # or
    cd manuscript/scripts/scratch && python3 check_unicode_math.py
"""

import os
import sys
import glob
import re

UNICODE_MATH_CHARS = [
    'ℝ', 'ℤ', 'ℕ', '∈', '∉', '⊂', '⊆', '∀', '∃', '∇', '∂',
    'σ', 'μ', 'λ', 'β', 'γ', 'θ', 'ω', 'ν', 'τ', 'ρ', 'ϕ', 'φ', 'ψ', 'α',
    '∑', '∏', '∫', '∬', '∭', '∮', '√', '∞', '≈', '≠', '≤', '≥', '±', '×', '÷',
    '⊗', '⊕', 'ᵀ', '′', '″', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉', '₀',
    '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹', '⁰'
]

META_DOCS = {'README.md', 'TODO.md', 'antigravity_notes.md', 'notation_index.md'}

def find_manuscript_dir():
    if os.path.isdir('manuscript'):
        return os.path.abspath('manuscript')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if os.path.isdir(candidate) and any(f.endswith('.md') for f in os.listdir(candidate)):
        return candidate
    return os.path.abspath('.')

def check_unicode_math(include_meta_docs=False):
    manuscript_dir = find_manuscript_dir()
    md_files = sorted(glob.glob(os.path.join(manuscript_dir, '*.md')))
    
    total_anomalies = 0
    checked_files = 0
    meta_doc_anomalies = 0
    
    print(f"Checking for raw Unicode math symbols across manuscript files...")
    print(f"Directory: {manuscript_dir}")
    print("-" * 70)
    
    for f in md_files:
        basename = os.path.basename(f)
        is_meta = basename in META_DOCS
        
        if is_meta and not include_meta_docs:
            # Check meta doc just to tally instructional mentions
            with open(f, 'r', encoding='utf-8') as fh:
                text = fh.read()
            text_no_code = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
            for line in text_no_code.split('\n'):
                for char in UNICODE_MATH_CHARS:
                    if char in line:
                        meta_doc_anomalies += 1
            continue
            
        checked_files += 1
        with open(f, 'r', encoding='utf-8') as fh:
            text = fh.read()
            
        # Strip code blocks
        text_no_code = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        
        file_anomalies = []
        for idx, line in enumerate(text_no_code.split('\n'), 1):
            for char in UNICODE_MATH_CHARS:
                if char in line:
                    file_anomalies.append((idx, char, line.strip()))
                    
        if file_anomalies:
            print(f"  [FOUND UNICODE] {basename}: {len(file_anomalies)} instances")
            for line_no, char, snippet in file_anomalies[:3]:
                print(f"     L{line_no} ('{char}'): {snippet[:80]}")
            total_anomalies += len(file_anomalies)
            
    print("-" * 70)
    print(f"Monograph Content Files Checked: {checked_files} (Chapters, Workbooks, Appendices)")
    print(f"Content File Raw Unicode Math Anomalies: {total_anomalies}")
    if not include_meta_docs:
        print(f"Meta-Documentation Instructional Mentions (whitelisted): {meta_doc_anomalies}")
    passed = (total_anomalies == 0)
    print(f"Status: {'PASSED (Zero raw Unicode math in monograph content)' if passed else 'FAILED'}")
    return passed

if __name__ == '__main__':
    inc_meta = '--include-meta' in sys.argv
    ok = check_unicode_math(include_meta_docs=inc_meta)
    sys.exit(0 if ok else 1)
