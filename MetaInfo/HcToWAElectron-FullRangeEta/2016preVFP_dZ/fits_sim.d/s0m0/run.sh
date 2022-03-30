#!/bin/bash
cd $TNP_BASE
python tnp_tamsa.py config/HcToWAElectron.py 2016preVFP_dZ --step fit --set 0 --member 0 --sim --bin $1 --no-condor
exit $?
