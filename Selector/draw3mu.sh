#!/bin/zsh

vars=('muons/1/pt'  'muons/2/pt'  'muons/3/pt' 
	  'muons/1/eta' 'muons/2/eta' 'muons/3/eta' 
	  'muons/1/phi' 'muons/2/phi' 'muons/3/phi'
	  'jets/1/pt'   'jets/1/eta'  'jets/1/phi'
	  'jets/2/pt'   'jets/2/pt'   'jets/2/phi'
	  'jets/Nj'     'ZMass'		  'MET'       
	  'nPV'         'nPileUp')

for var in "${vars[@]}"
do
	python Draw3Mu.py --histkey $var
done
