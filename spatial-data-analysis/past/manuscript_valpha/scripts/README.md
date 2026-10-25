# Monograph Scripts & Tooling Directory

This directory houses the computational tooling and quality assurance infrastructure for the graduate monograph:
**"Theoretical Foundations of Spatial Statistics and Geostatistics: A Rigorous Synthesis and Computational Companion"**

## Quick Start

All active production tools, verification test suites, and academic peer-review records are located in:
👉 **[`manuscript/scripts/scratch/`](scratch/README.md)**

To run the complete verification suite across all 34 monograph documents:
```bash
python3 manuscript/scripts/scratch/verify_all.py
```

To view corpus metrics and scale:
```bash
python3 manuscript/scripts/scratch/audit_corpus_sizes.py
```

To concatenate and compile the full monograph PDF:
```bash
python3 manuscript/scripts/build_pdf.py
```


For complete documentation on each tool, design philosophy, and failure recovery procedures, see **[`scratch/README.md`](scratch/README.md)**.
