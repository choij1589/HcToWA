#!/bin/sh
python Analyzer.py --sample DATA --region DY --channel 3Mu &
python Analyzer.py --sample DY --region DY --channel 3Mu &
python Analyzer.py --sample ttX --region DY --channel 3Mu &
python Analyzer.py --sample rare --region DY --channel 3Mu &
python Analyzer.py --sample VV --region DY --channel 3Mu &

python Analyzer.py --sample DATA --region ZG --channel 3Mu &
python Analyzer.py --sample ZG --region ZG --channel 3Mu &
python Analyzer.py --sample ttX --region ZG --channel 3Mu &
python Analyzer.py --sample rare --region ZG --channel 3Mu &
python Analyzer.py --sample VV --region ZG --channel 3Mu &
