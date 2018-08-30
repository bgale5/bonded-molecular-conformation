#!/usr/bin/env bash

NUM_GENS=100

for i in {3..61};
do
    echo "Beginning N = $i"
    filepath=documentation/without_sa/n$i
    mkdir -p $filepath
    cmake-build-release/bonded_molecular_conformation $i $NUM_GENS | tee $filepath/results.csv | ./plot.py $filepath
done
