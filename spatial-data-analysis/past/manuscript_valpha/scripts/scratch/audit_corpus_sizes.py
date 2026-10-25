#!/usr/bin/env python3
"""
audit_corpus_sizes.py
---------------------
Computes line counts, byte sizes, and cumulative corpus metrics across all
34 Markdown files comprising the graduate monograph.

Provides a structured overview of the monograph's physical scale.

Usage:
    python3 manuscript/scripts/scratch/audit_corpus_sizes.py
    # or
    cd manuscript/scripts/scratch && python3 audit_corpus_sizes.py
"""

import os
import sys
import glob

def find_manuscript_dir():
    if os.path.isdir('manuscript'):
        return os.path.abspath('manuscript')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(script_dir, '..', '..'))
    if os.path.isdir(candidate) and any(f.endswith('.md') for f in os.listdir(candidate)):
        return candidate
    return os.path.abspath('.')

def audit_corpus_sizes():
    manuscript_dir = find_manuscript_dir()
    md_files = sorted(glob.glob(os.path.join(manuscript_dir, '*.md')))
    
    print("=" * 78)
    print(f"{'Filename':<42} | {'Lines':>8} | {'Bytes':>10} | {'Size (KB)':>10}")
    print("=" * 78)
    
    total_lines = 0
    total_bytes = 0
    
    for f in md_files:
        basename = os.path.basename(f)
        with open(f, 'r', encoding='utf-8') as fh:
            lines = len(fh.readlines())
        bytes_len = os.path.getsize(f)
        kb = bytes_len / 1024.0
        total_lines += lines
        total_bytes += bytes_len
        print(f"{basename:<42} | {lines:8d} | {bytes_len:10d} | {kb:9.1f} KB")
        
    print("-" * 78)
    total_mb = total_bytes / (1024.0 * 1024.0)
    print(f"{'TOTAL CORPUS (' + str(len(md_files)) + ' files)':<42} | {total_lines:8d} | {total_bytes:10d} | {total_mb:9.2f} MB")
    print("=" * 78)
    return total_lines, total_bytes

if __name__ == '__main__':
    audit_corpus_sizes()
    sys.exit(0)
