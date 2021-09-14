#!/bin/sh
CHANNEL=$1

# mass points
#MHc160=(15 45 75 85 120 155)
#MHc130=(15 45 55 90 125)
#MHc100=(15 25 60 95)
#MHc70=(15 40 65)
MHc160=(120 155)
MHc130=()
MHc100=()
MHc70=()

for MA in "${MHc70[@]}"
do
  MHc=70
  python Analyzer.py --sample DATA --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample DY --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample ZG --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample rare --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample VV --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample TTToHcToWA_AToMuMu_MHc${MHc}_MA${MA} --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample ttX --channel ${CHANNEL} --mass MHc${MHc}_MA${MA}
done

for MA in "${MHc100[@]}"
do
  MHc=100
  python Analyzer.py --sample DATA --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample DY --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample ZG --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample rare --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample VV --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample TTToHcToWA_AToMuMu_MHc${MHc}_MA${MA} --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample ttX --channel ${CHANNEL} --mass MHc${MHc}_MA${MA}
done

for MA in "${MHc130[@]}"
do
  MHc=130
  python Analyzer.py --sample DATA --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample DY --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample ZG --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample rare --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample VV --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample TTToHcToWA_AToMuMu_MHc${MHc}_MA${MA} --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample ttX --channel ${CHANNEL} --mass MHc${MHc}_MA${MA}
done

for MA in "${MHc160[@]}"
do
  MHc=160
  python Analyzer.py --sample DATA --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample DY --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample ZG --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample rare --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample VV --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample TTToHcToWA_AToMuMu_MHc${MHc}_MA${MA} --channel ${CHANNEL} --mass MHc${MHc}_MA${MA} &
  python Analyzer.py --sample ttX --channel ${CHANNEL} --mass MHc${MHc}_MA${MA}
done
