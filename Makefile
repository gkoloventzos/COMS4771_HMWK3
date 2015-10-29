assignment=assignment
LATEX=pdflatex

SRCS=$(wildcard *.tex)
REFS=$(wildcard *.bib)
FIGS=$(wildcard figures/*)

.PHONY: $(assignment).pdf clean

all: $(assignment).pdf

$(assignment).pdf: $(SRCS) $(REFS) $(FIGS)
	$(LATEX) $(assignment)
	$(LATEX) $(assignment)


clean:
	rm -f *.dvi *.aux *.log *.blg *.bbl \
          $(assignment).ps $(assignment).pdf \
          $(assignment)-single.pdf $(assignment)-single.ps
