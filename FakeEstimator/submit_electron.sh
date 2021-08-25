#!/bin/sh
python3 submit.py -o electron -s SingleElectron -n 10
python3 submit.py -o electron -s QCD_EMEnriched -n 10
python3 submit.py -o electron -s QCD_bcToE      -n 10
python3 submit.py -o electron -s W              -n 15
python3 submit.py -o electron -s DY             -n 50
python3 submit.py -o electron -s TT             -n 50
python3 submit.py -o electron -s ST             -n 15
python3 submit.py -o electron -s VV             -n 10
