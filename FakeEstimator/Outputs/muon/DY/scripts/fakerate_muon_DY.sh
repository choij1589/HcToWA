#!/bin/sh
cd /data6/Users/choij/CMSSW/CMSSW_11_3_0_pre2/src/HcToWA/FakeEstimator
process=$1

source /cvmfs/cms.cern.ch/cmsset_default.sh
eval `scramv1 runtime -sh`
python3 Analyzer.py -s DY -o muon -f syst -n 40 -t ${process}