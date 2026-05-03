MAIN = main
NEURIPS = main_neurips

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex appendix/*.tex references.bib
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

neurips: $(NEURIPS).pdf

$(NEURIPS).pdf: $(NEURIPS).tex references.bib figures/*.pdf
	pdflatex $(NEURIPS)
	bibtex $(NEURIPS)
	pdflatex $(NEURIPS)
	pdflatex $(NEURIPS)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk

view: $(MAIN).pdf
	open $(MAIN).pdf

view-neurips: $(NEURIPS).pdf
	open $(NEURIPS).pdf
