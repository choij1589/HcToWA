#!/bin/bash
cd $TNP_BASE
python tnp_tamsa.py config/HcToWAElectron.py 2017_HoverE --step fit --set 0 --member 0 --data --bin $1 --no-condor
exit $?
