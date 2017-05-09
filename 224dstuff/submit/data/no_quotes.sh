#!/bin/sh

for f in *.tsv; do
    cat $f | sed 's/"//g' > ${f}.2
    mv ${f}.2 $f
done
