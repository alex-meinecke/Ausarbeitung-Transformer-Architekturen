#!/bin/bash

pdflatex -synctex=1 -interaction=nonstopmode Praxisbericht.tex
biber Praxisbericht
pdflatex -synctex=1 -interaction=nonstopmode Praxisbericht.tex
pdflatex -synctex=1 -interaction=nonstopmode Praxisbericht.tex
