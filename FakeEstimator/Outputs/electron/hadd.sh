#!/bin/zsh

SAMPLES=(SingleElectron QCD_EMEnriched QCD_bcToE W DY TT ST VV)

for sample in ${SAMPLES[@]};
do
	hadd SglEle_${sample}.root ${sample}/*.root
done
