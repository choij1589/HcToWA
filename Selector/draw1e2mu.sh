#!/bin/zsh

vars=('muons/1/pt'  'muons/2/pt'  'electrons/1/pt' 
	  'muons/1/eta' 'muons/2/eta' 'electrons/1/eta' 
	  'muons/1/phi' 'muons/2/phi' 'electrons/1/phi'
	  'jets/1/pt'   'jets/1/eta'  'jets/1/phi'
	  'jets/2/pt'   'jets/2/pt'   'jets/2/phi'
	  'jets/Nj'     'ZMass'        'MET'     
	  'nPV'         'nPileUp')

for var in "${vars[@]}"
do
	python Draw1E2Mu.py --histkey $var
done
