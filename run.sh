#!/usr/bin/env bash

cmake-build-release/bonded_molecular_conformation | tee documentation/results.csv | ./plot.py
