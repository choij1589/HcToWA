#!/bin/sh
ERAs=( 2016preVFP 2016postVFP 2017 2018 )

for ERA in "${ERAs[@]}"
do
	echo drawing plots for ${ERA}...
	mkdir -p plots/${ERA}/MuonIDSFOnly
	python drawPlots.py -e ${ERA} -v diMuon/pt -o plots/${ERA}/MuonIDSFOnly/diMuon_pt.png
	python drawPlots.py -e ${ERA} -v diMuon/eta -o plots/${ERA}/MuonIDSFOnly/diMuon_eta.png
	python drawPlots.py -e ${ERA} -v diMuon/phi -o plots/${ERA}/MuonIDSFOnly/diMuon_phi.png
	python drawPlots.py -e ${ERA} -v diMuon/mass -o plots/${ERA}/MuonIDSFOnly/diMuon_mass.png
	python drawPlots.py -e ${ERA} -v muons/1/pt -o plots/${ERA}/MuonIDSFOnly/muon1_pt.png
	python drawPlots.py -e ${ERA} -v muons/1/eta -o plots/${ERA}/MuonIDSFOnly/muon1_eta.png
	python drawPlots.py -e ${ERA} -v muons/1/phi -o plots/${ERA}/MuonIDSFOnly/muon1_phi.png
	python drawPlots.py -e ${ERA} -v muons/2/pt -o plots/${ERA}/MuonIDSFOnly/muon2_pt.png
    python drawPlots.py -e ${ERA} -v muons/2/eta -o plots/${ERA}/MuonIDSFOnly/muon2_eta.png
    python drawPlots.py -e ${ERA} -v muons/2/phi -o plots/${ERA}/MuonIDSFOnly/muon2_phi.png
	python drawPlots.py -e ${ERA} -v jets/1/pt -o plots/${ERA}/MuonIDSFOnly/j1_pt.png
	python drawPlots.py -e ${ERA} -v jets/1/eta -o plots/${ERA}/MuonIDSFOnly/j1_eta.png
	python drawPlots.py -e ${ERA} -v jets/1/phi -o plots/${ERA}/MuonIDSFOnly/j1_phi.png
	python drawPlots.py -e ${ERA} -v jets/2/pt -o plots/${ERA}/MuonIDSFOnly/j2_pt.png
	python drawPlots.py -e ${ERA} -v jets/2/eta -o plots/${ERA}/MuonIDSFOnly/j2_eta.png
	python drawPlots.py -e ${ERA} -v jets/2/phi -o plots/${ERA}/MuonIDSFOnly/j2_phi.png
	python drawPlots.py -e ${ERA} -v jets/size -o plots/${ERA}/MuonIDSFOnly/jet_multiplicity.png
	python drawPlots.py -e ${ERA} -v jets/HT -o plots/${ERA}/MuonIDSFOnly/HT.png
done

