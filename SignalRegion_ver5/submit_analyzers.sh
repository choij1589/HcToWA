#!/bin/sh
CHANNEL=$1
MASS_POINT=$2

python Utilities/submitAnayzerToCondor.py -s DATA -c $CHANNEL -m ${MASS_POINT}
python Utilities/submitAnayzerToCondor.py -s DY -c $CHANNEL -m ${MASS_POINT}
python Utilities/submitAnayzerToCondor.py -s ZG -c $CHANNEL -m ${MASS_POINT}
python Utilities/submitAnayzerToCondor.py -s ttX -c $CHANNEL -m ${MASS_POINT}
python Utilities/submitAnayzerToCondor.py -s rare -c $CHANNEL -m ${MASS_POINT}
python Utilities/submitAnayzerToCondor.py -s VV -c $CHANNEL -m ${MASS_POINT}
python Utilities/submitAnayzerToCondor.py -s TTToHcToWA_AToMuMu_${MASS_POINT} -c $CHANNEL -m ${MASS_POINT}
