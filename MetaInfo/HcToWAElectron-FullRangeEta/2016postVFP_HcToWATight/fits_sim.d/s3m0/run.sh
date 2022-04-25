#!/bin/bash
cd $TNP_BASE
python tnp_tamsa.py config/HcToWAElectron.py 2016postVFP_HcToWATight --step fit --set 3 --member 0 --sim --bin $1 --no-condor
exit $?
