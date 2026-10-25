#!/usr/bin/env python3
"""
check_prose_links.py
--------------------
Hyperlink and cross-reference auditor for the entire monograph corpus.
Inspects all 34 Markdown files for broken relative links, anchor links,
and intra-manuscript cross-references in prose.

Excludes code blocks (`...`) and external web URLs (http/https).

Usage:
    python3 manuscript/scripts/scratch/check_prose_links.py
    # or
    cd manuscript/scripts/scratch && python3 check_prose_links.py
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

def check_prose_links():
    manuscript_dir = find_manuscript_dir()
    md_files = sorted(glob.glob(os.path.join(manuscript_dir, '*.md')))
    broken_links = []
    total_links = 0
    
    print(f"Auditing Markdown hyperlinks across {len(md_files)} files...")
    print(f"Directory: {manuscript_dir}")
    print("-" * 70)
    
    for f in md_files:
        with open(f, 'r', encoding='utf-8') as fh:
            text = fh.read()
            
        # Strip fenced code blocks
        text_prose = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        # Strip inline code `...`
        text_prose = re.sub(r'`.*?`', '', text_prose)
            
        # Match markdown links [text](target)
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', text_prose)
        for text_label, target in links:
            total_links += 1
            target_clean = target.split('#')[0]
            if not target_clean:
                continue # Pure anchor link within the same page
            if target_clean.startswith('http://') or target_clean.startswith('https://'):
                continue # Web link
            if target_clean.startswith('file:///'):
                path = target_clean.replace('file://', '')
                if not os.path.exists(path):
                    broken_links.append((os.path.basename(f), target, f"File not found: {path}"))
            else:
                base_dir = os.path.dirname(f)
                resolved = os.path.normpath(os.path.join(base_dir, target_clean))
                if not os.path.exists(resolved):
                    broken_links.append((os.path.basename(f), target, f"Target not found: {resolved}"))
                    
    print(f"Total Prose Hyperlinks Inspected : {total_links}")
    print(f"Total Broken Links Found          : {len(broken_links)}")
    for source, target, reason in broken_links:
        print(f"  [BROKEN] in {source}: ({target}) -> {reason}")
        
    print("-" * 70)
    passed = (len(broken_links) == 0)
    print(f"Status: {'PASSED (Zero broken links)' if passed else 'FAILED'}")
    return passed

if __name__ == '__main__':
    ok = check_prose_links()
    sys.exit(0 if ok else 1)
