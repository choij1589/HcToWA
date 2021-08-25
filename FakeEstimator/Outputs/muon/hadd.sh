#!/bin/zsh

SAMPLES=(DoubleMuon QCD_MuEnriched W DY TT ST VV)

for sample in ${SAMPLES[@]};
do
	hadd SglMu_${sample}.root ${sample}/*.root
done
