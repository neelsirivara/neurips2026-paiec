#!/usr/bin/env bash
# Regenerate every number, table and figure of the report from experiments/results/ and compile it.
#   bash experiments/src/build_paper.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PAPER="$HERE/../../paper"
cd "$HERE" || exit 1
python make_macros.py || exit 1
python make_figures.py || exit 1
python make_audit.py || exit 1
cd "$PAPER" || exit 1
rm -f main.aux main.bbl main.blg main.out
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build1.log 2>&1 || { grep -A5 "^! " build1.log | head -20; exit 1; }
bibtex main > bib.log 2>&1 || { cat bib.log; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build2.log 2>&1 || { grep -A5 "^! " build2.log | head -20; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build3.log 2>&1 || { grep -A5 "^! " build3.log | head -20; exit 1; }
pdftotext -layout main.pdf main_text.txt
echo "pages: $(pdfinfo main.pdf | grep Pages | awk '{print $2}')"
echo "undefined refs/citations: $(grep -c 'undefined' build3.log)"
echo "overfull boxes: $(grep -c 'Overfull' build3.log)"
echo "'??' in PDF text: $(grep -c '??' main_text.txt)"
echo "'TODO' in PDF text: $(grep -ci 'TODO' main_text.txt)"
echo "'REVISE' in tex: $(grep -c 'REVISE' main.tex)"
grep -n "??" main_text.txt | head -10
