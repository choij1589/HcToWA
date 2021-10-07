#!/bin/sh
CHANNEL=$1

# mass points
MASS_POINTS=(MHc70_MA15 MHc70_MA40 MHc70_MA65
             MHc100_MA15 MHc100_MA25 MHc100_MA60 MHc100_MA95
             MHc130_MA15 MHc130_MA45 MHc130_MA55 MHc130_MA90 MHc130_MA125
             MHc160_MA15 MHc160_MA45 MHc160_MA75 MHc160_MA85 MHc160_MA120 MHc160_MA155)
for mp in "${MASS_POINTS[@]}"
do
	python3 prepare_datacard.py --channel ${CHANNEL} --mass_point ${mp} --grid 1D
	python3 prepare_datacard.py --channel ${CHANNEL} --mass_point ${mp} --grid 3D
	combine -M AsymptoticLimits datacard_${CHANNEL}_${mp}_1D.dat \
	-n .${CHANNEL}_${mp}_1D --bypassFrequentistFit -t -1\
	--noFitAsimov --run expected >> ${CHANNEL}_${mp}_1D.txt
	
	combine -M AsymptoticLimits datacard_${CHANNEL}_${mp}_3D.dat \
	-n .${CHANNEL}_${mp}_3D --bypassFrequentistFit -t -1 \
	--noFitAsimov --run expected >> ${CHANNEL}_${mp}_3D.txt
done
