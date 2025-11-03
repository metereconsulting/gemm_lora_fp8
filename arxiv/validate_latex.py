#!/usr/bin/env python3
"""
Simple LaTeX validation script for the arXiv paper.
Checks for basic syntax issues and citation consistency.
"""

import re
import sys

def extract_citations(text):
    """Extract all citation keys from \cite{} commands."""
    cite_pattern = r'\\cite\{([^}]+)\}'
    citations = []
    for match in re.finditer(cite_pattern, text):
        keys = match.group(1).split(',')
        citations.extend([key.strip() for key in keys])
    return set(citations)

def extract_bib_keys(bib_text):
    """Extract all BibTeX keys from the .bib file."""
    key_pattern = r'@\w+\{([^,]+),'
    keys = []
    for match in re.finditer(key_pattern, bib_text):
        keys.append(match.group(1))
    return set(keys)

def check_braces(text):
    """Check for unmatched braces."""
    brace_count = 0
    for char in text:
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count < 0:
                return False
    return brace_count == 0

def main():
    print("Validating LaTeX paper for arXiv submission...")

    # Read files
    try:
        with open('paper.tex', 'r') as f:
            tex_content = f.read()
        with open('references.bib', 'r') as f:
            bib_content = f.read()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Check brace matching
    if not check_braces(tex_content):
        print("Warning: Unmatched braces in paper.tex")
    else:
        print("✓ Brace matching OK")

    # Extract citations and bib keys
    citations = extract_citations(tex_content)
    bib_keys = extract_bib_keys(bib_content)

    print(f"Found {len(citations)} citations in paper.tex")
    print(f"Found {len(bib_keys)} references in references.bib")

    # Check for missing references
    missing_refs = citations - bib_keys
    if missing_refs:
        print(f"❌ Missing references: {sorted(missing_refs)}")
        return 1
    else:
        print("✓ All citations have corresponding references")

    # Check for unused references
    unused_refs = bib_keys - citations
    if unused_refs:
        print(f"⚠️  Unused references: {sorted(unused_refs)}")
    else:
        print("✓ All references are used")

    # Check for common LaTeX issues
    issues = []

    # Check for double dollar signs
    if '$$' in tex_content:
        issues.append("Found $$ (display math) - consider using \\[\\] instead")

    # Check for common typos
    if '\\begin{document}' not in tex_content:
        issues.append("Missing \\begin{document}")

    if '\\end{document}' not in tex_content:
        issues.append("Missing \\end{document}")

    if '\\title{' not in tex_content:
        issues.append("Missing \\title{}")

    if '\\author{' not in tex_content:
        issues.append("Missing \\author{}")

    if '\\maketitle' not in tex_content:
        issues.append("Missing \\maketitle")

    if issues:
        print("\nLaTeX issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    else:
        print("✓ No obvious LaTeX issues found")

    print("\n✅ LaTeX validation passed!")
    print("\nTo compile the PDF, run one of these commands:")
    print("  pdflatex paper.tex")
    print("  pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex")
    print("\nOr use an online LaTeX compiler like Overleaf.")

    return 0

if __name__ == '__main__':
    sys.exit(main())
