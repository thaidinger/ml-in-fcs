PAPER = main

.PHONY: all clean watch

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex sections/*.tex references.bib
	pdflatex -interaction=nonstopmode $(PAPER).tex
	bibtex $(PAPER) || true
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex

clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.fls *.fdb_latexmk *.synctex.gz sections/*.aux

distclean: clean
	rm -f $(PAPER).pdf

# Live rebuild while editing — requires latexmk
watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(PAPER).tex
