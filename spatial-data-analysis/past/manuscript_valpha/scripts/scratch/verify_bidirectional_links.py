#!/usr/bin/env python3
"""
verify_bidirectional_links.py
-----------------------------
Validates that every monograph Chapter text file (ch01 through ch13) explicitly
links to its companion Exercise Workbook, and conversely, every companion
Exercise Workbook links back to its parent Chapter text file.

Ensures zero orphaned chapters or workbooks across the monograph.

Usage:
    python3 manuscript/scripts/scratch/verify_bidirectional_links.py
    # or
    cd manuscript/scripts/scratch && python3 verify_bidirectional_links.py
"""

import os
import sys

def find_manuscript_dir():
    if os.path.isdir('manuscript'):
        return os.path.abspath('manuscript')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if os.path.isdir(candidate) and any(f.endswith('.md') for f in os.listdir(candidate)):
        return candidate
    return os.path.abspath('.')

def verify_chapter_workbook_bidirectional_links():
    manuscript_dir = find_manuscript_dir()
    print("Verifying bidirectional links between chapters and workbooks...")
    print(f"Directory: {manuscript_dir}")
    print("-" * 70)
    
    missing_links = []
    checked_pairs = 0
    
    for ch_num in range(1, 14):
        ch_str = f"{ch_num:02d}"
        ch_file = None
        for f in os.listdir(manuscript_dir):
            if f.startswith(f"ch{ch_str}_") and not f.endswith("_exercises.md") and f.endswith(".md"):
                ch_file = os.path.join(manuscript_dir, f)
                break
        wb_file = os.path.join(manuscript_dir, f"ch{ch_str}_exercises.md")
        
        if not ch_file or not os.path.exists(ch_file):
            missing_links.append(f"Chapter {ch_str} text file not found!")
            continue
        if not os.path.exists(wb_file):
            missing_links.append(f"Workbook {wb_file} not found!")
            continue
            
        checked_pairs += 1
        
        # Check if chapter text references wb_file
        with open(ch_file, 'r', encoding='utf-8') as fh:
            ch_content = fh.read()
        wb_basename = os.path.basename(wb_file)
        if wb_basename not in ch_content:
            missing_links.append(f"Chapter {os.path.basename(ch_file)} does not link to workbook {wb_basename}")
            
        # Check if workbook references ch_file
        with open(wb_file, 'r', encoding='utf-8') as fh:
            wb_content = fh.read()
        ch_basename = os.path.basename(ch_file)
        if ch_basename not in wb_content:
            missing_links.append(f"Workbook {wb_basename} does not link to chapter {ch_basename}")
            
    print(f"Checked Chapter-Workbook Pairs: {checked_pairs} / 13")
    print(f"Total Missing Bidirectional Links: {len(missing_links)}")
    for err in missing_links:
        print(f"  [MISSING LINK] {err}")
    print("-" * 70)
    passed = (len(missing_links) == 0 and checked_pairs == 13)
    print(f"Status: {'PASSED (All 13 Chapter <-> Workbook links verified)' if passed else 'FAILED'}")
    return passed

if __name__ == '__main__':
    ok = verify_chapter_workbook_bidirectional_links()
    sys.exit(0 if ok else 1)
