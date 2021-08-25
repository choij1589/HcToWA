#!/bin/sh

# mass points
MHc160=(15 45 75 85 120 155)
MHc130=(15 45 55 90 125)
MHc100=(15 25 60 95)
MHc70=(15 40 65)

for MA in "${MHc160[@]}"
do
  MHc=160
  python Analyzer.py --sample DATA --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample DY --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample ZG --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample ttX --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample rare --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample VV --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample TTToHcToWA_AToMuMu_MHc${MHc}_MA${MA} --channel 3Mu --mass MHc${MHc}_MA${MA}
done

for MA in "${MHc130[@]}"
do
  MHc=130
  python Analyzer.py --sample DATA --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample DY --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample ZG --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample ttX --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample rare --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample VV --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample TTToHcToWA_AToMuMu_MHc${MHc}_MA${MA} --channel 3Mu --mass MHc${MHc}_MA${MA}
done

for MA in "${MHc100[@]}"
do
  MHc=100
  python Analyzer.py --sample DATA --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample DY --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample ZG --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample ttX --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample rare --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample VV --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample TTToHcToWA_AToMuMu_MHc${MHc}_MA${MA} --channel 3Mu --mass MHc${MHc}_MA${MA}
done

for MA in "${MHc70[@]}"
do
  MHc=70
  python Analyzer.py --sample DATA --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample DY --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample ZG --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample ttX --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample rare --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample VV --channel 3Mu --mass MHc${MHc}_MA${MA}
  python Analyzer.py --sample TTToHcToWA_AToMuMu_MHc${MHc}_MA${MA} --channel 3Mu --mass MHc${MHc}_MA${MA}
done
