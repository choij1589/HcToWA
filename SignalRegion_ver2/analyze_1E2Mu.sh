#!/bin/sh

MHc=130
MA=90

python Analyzer.py --sample DATA --channel 1E2Mu --mass MHc${MHc}_MA${MA}
python Analyzer.py --sample DY --channel 1E2Mu --mass MHc${MHc}_MA${MA}
python Analyzer.py --sample ZG --channel 1E2Mu --mass MHc${MHc}_MA${MA}
python Analyzer.py --sample ttX --channel 1E2Mu --mass MHc${MHc}_MA${MA}
python Analyzer.py --sample rare --channel 1E2Mu --mass MHc${MHc}_MA${MA}
python Analyzer.py --sample VV --channel 1E2Mu --mass MHc${MHc}_MA${MA}
python Analyzer.py --sample TTToHcToWA_AToMuMu_MHc${MHc}_MA${MA} --channel 1E2Mu --mass MHc${MHc}_MA${MA}