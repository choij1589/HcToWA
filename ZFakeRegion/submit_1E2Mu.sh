#!/bin/sh
python Analyzer.py --sample DATA --channel 1E2Mu &
python Analyzer.py --sample DY --channel 1E2Mu &
python Analyzer.py --sample ZG --channel 1E2Mu &
python Analyzer.py --sample ttX --channel 1E2Mu &
python Analyzer.py --sample rare --channel 1E2Mu &
python Analyzer.py --sample VV --channel 1E2Mu &
