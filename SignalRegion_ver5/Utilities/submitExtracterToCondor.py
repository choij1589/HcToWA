import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--channel", "-c", default=None, required=True, type=str, help='channel')
args = parser.parse_args()
CHANNEL = args.channel

MassPoints = {70: [15, 40, 65],
              100: [15, 25, 60, 95],
              130: [15, 45, 55, 90, 125],
              160: [15, 45, 75, 85, 120, 155]}
# MassPoints = {70: [15]}
if not os.path.exists(f"condor_base/extraction/{CHANNEL}"):
	os.makedirs(f"condor_base/extraction/{CHANNEL}")
os.chdir(f"condor_base/extraction/{CHANNEL}/")

for MHc in MassPoints.keys():
	for MA in MassPoints[MHc]:
		with open(f"extract_MHc-{MHc}_MA-{MA}.sub", "w") as f:
			f.write(f"""executable				= extract_MHc-{MHc}_MA-{MA}.sh
arguments					= $(ClusterId)$(ProcId)
output						= extract_MHc-{MHc}_MA-{MA}.$(ClusterId).$(ProcId).out
error						= extract_MHc-{MHc}_MA-{MA}.$(ClusterId).$(ProcId).err
log							= extract_MHc-{MHc}_MA-{MA}.$(ClusterId).$(ProcId).log
should_transfer_files		= Yes
when_to_transfer_output		= ON_EXIT
request_memory				= 10000
queue""")

		with open(f"extract_MHc-{MHc}_MA-{MA}.sh", "w") as f:
			f.write(f"""#!/bin/sh
source /u/user/choij/scratch/miniconda3/bin/activate
conda activate torch

cd /u/user/choij/scratch/HcToWA/SignalRegion_ver5
python signal_extraction.py --mass MHc{MHc}_MA{MA} --channel {CHANNEL}
""")
		os.system(f"condor_submit extract_MHc-{MHc}_MA-{MA}.sub")
