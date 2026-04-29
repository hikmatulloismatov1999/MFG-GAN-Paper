MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex appendix/*.tex references.bib
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk

view: $(MAIN).pdf
	open $(MAIN).pdf
