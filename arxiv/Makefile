# Makefile for compiling the arXiv paper

.PHONY: all clean pdf validate help

# Default target
all: pdf

# Compile PDF with bibliography
pdf: paper.tex references.bib
	pdflatex paper.tex
	bibtex paper
	pdflatex paper.tex
	pdflatex paper.tex

# Quick compile without bibliography (for syntax checking)
quick: paper.tex
	pdflatex paper.tex

# Validate LaTeX syntax and references
validate:
	python3 validate_latex.py

# Clean up auxiliary files
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fdb_latexmk *.fls

# Clean up all generated files including PDF
cleanall: clean
	rm -f paper.pdf

# Help target
help:
	@echo "Available targets:"
	@echo "  all      - Compile PDF with full bibliography processing (default)"
	@echo "  pdf      - Same as 'all'"
	@echo "  quick    - Quick compile without bibliography (for syntax checking)"
	@echo "  validate - Run LaTeX validation script"
	@echo "  clean    - Remove auxiliary files (*.aux, *.bbl, etc.)"
	@echo "  cleanall - Remove auxiliary files and PDF"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make              # Compile full PDF"
	@echo "  make validate     # Check for errors before compiling"
	@echo "  make clean && make # Clean and recompile"
