#!/bin/sh

python Analyzer.py -s DATA -c 1E2Mu -m MHc160_MA155 &
python Analyzer.py -s DY -c 1E2Mu -m MHc160_MA155 &
python Analyzer.py -s ZG -c 1E2Mu -m MHc160_MA155 &
python Analyzer.py -s ttX -c 1E2Mu -m MHc160_MA155 &
python Analyzer.py -s rare -c 1E2Mu -m MHc160_MA155 &
python Analyzer.py -s VV -c 1E2Mu -m MHc160_MA155 &
python Analyzer.py -s TTToHcToWA_AToMuMu_MHc160_MA155 -c 1E2Mu -m MHc160_MA155 &
