# Author: Jin Choi
# Date: Oct 7, 2021
# Script to prepare the datacards
# as an input to the Higgs Combine Tools
# python3 prepare_datacard.py --channel $CHANNEL --mass_point $MASS_POINT --grid $GRID
import os, argparse
import numpy as np
import pandas as pd

# parse inputs and get environment variables
parser = argparse.ArgumentParser()
parser.add_argument("--channel", "-c", default=None, required=True, type=str, help="channel")
parser.add_argument("--mass_point", "-m", default=None, required=True, type=str, help="signal mass point")
parser.add_argument("--grid", "-g", default="1D", required=False, type=str)
args = parser.parse_args()

CHANNEL = args.channel			# 1E2Mu or 3Mu
MASS_POINT = args.mass_point	# mass points
GRID = args.grid				# 1D or 3D
HcToWA_DIR = os.environ['WORKING_DIR']

# read dataframe
df = pd.read_csv(f"{HcToWA_DIR}/SignalRegion_ver5/Outputs/{CHANNEL}/CSV/Grid_{MASS_POINT}_exact_{GRID}.csv")
df = df.set_index("Unnamed: 0")[['signal', 'fake', 'conv', 'ttX', 'VV', 'rare']]

rates = df.values[0]
errors = df.values[1]

sig_rate = round(rates[0], 4)
bkg_rate = round(sum(rates[1:]), 4)

datacard = f"""imax          1 number of bins
jmax          1 number of processes minus 1
kmax          * number of nuisance parameters
--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
bin             SR
observation     10.0
--------------------------------------------------------------------------------------------------
bin                         SR                        SR
process	                    bkg                       signal
process	                    1                         0
rate                        {bkg_rate}                {sig_rate}
--------------------------------------------------------------------------------------------------
acceptance_fake    lnN      {1.+round(errors[1]/bkg_rate, 3)}   -
acceptance_conv    lnN      {1.+round(errors[2]/bkg_rate, 3)}   -
acceptance_ttX     lnN      {1.+round(errors[3]/bkg_rate, 3)}   -  
acceptance_VV      lnN      {1.+round(errors[4]/bkg_rate, 3)}   -
acceptance_rare    lnN      {1.+round(errors[5]/bkg_rate, 3)}   -
"""
print(datacard)
# save datacard
with open(f"datacard_{CHANNEL}_{MASS_POINT}_{GRID}.dat", "w") as f:
	f.write(datacard)

