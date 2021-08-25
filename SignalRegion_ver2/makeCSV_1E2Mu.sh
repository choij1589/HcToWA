#!/bin/sh
# mass points
MHc160=(15 45 75 85 120 155)
MHc130=(15 45 55 90 125)
MHc100=(15 25 60 95)
MHc70=(15 40 65)

python MakeCSV.py --sample DATA --channel 1E2Mu &
python MakeCSV.py --sample DY --channel 1E2Mu &
python MakeCSV.py --sample ZG --channel 1E2Mu &
python MakeCSV.py --sample ttX --channel 1E2Mu &
python MakeCSV.py --sample rare --channel 1E2Mu &
python MakeCSV.py --sample VV --channel 1E2Mu &
python MakeCSV.py --sample fake --channel 1E2Mu &

for MA in "${MHc160[@]}"
do
  MHc=160
  python MakeCSV.py --sample TTToHcToWA_AToMuMu_MHc${MHc}_MA${MA} --channel 1E2Mu &
done

for MA in "${MHc130[@]}"
do
  MHc=130
  python MakeCSV.py --sample TTToHcToWA_AToMuMu_MHc${MHc}_MA${MA} --channel 1E2Mu &
done

for MA in "${MHc100[@]}"
do
  MHc=100
  python MakeCSV.py --sample TTToHcToWA_AToMuMu_MHc${MHc}_MA${MA} --channel 1E2Mu &
done

for MA in "${MHc70[@]}"
do
  MHc=70
  python MakeCSV.py --sample TTToHcToWA_AToMuMu_MHc${MHc}_MA${MA} --channel 1E2Mu &
done
