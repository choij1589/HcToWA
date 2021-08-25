#!/bin/sh
python3 submit.py -o muon -s DoubleMuon -n 10
python3 submit.py -o muon -s QCD_MuEnriched -n 10
python3 submit.py -o muon -s W              -n 15
python3 submit.py -o muon -s DY             -n 40
python3 submit.py -o muon -s TT             -n 40
python3 submit.py -o muon -s ST             -n 15
python3 submit.py -o muon -s VV             -n 10
