import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--channel", "-c", default=None, required=True, type=str, help='channel')
parser.add_argument("--target", "-t", default=None, required=True, type=str, help='target')
args = parser.parse_args()
CHANNEL = args.channel
TARGET = args.target

MassPoints = {70: [15, 40, 65],
              100: [15, 25, 60, 95],
              130: [15, 45, 55, 90, 125],
              160: [15, 45, 75, 85, 120, 155]}
# MassPoints = {70: [15]}
os.chdir(f"condor_base/train/{CHANNEL}/{TARGET}")

for MHc in MassPoints.keys():
	for MA in MassPoints[MHc]:
		with open(f"train_MHc-{MHc}_MA-{MA}_vs_{TARGET}.sub", "w") as f:
			f.write(f"""executable				= train_MHc-{MHc}_MA-{MA}_vs_{TARGET}.sh
arguments					= $(ClusterId)$(ProcId)
output						= train_MHc-{MHc}_MA-{MA}_vs_{TARGET}.$(ClusterId).$(ProcId).out
error						= train_MHc-{MHc}_MA-{MA}_vs_{TARGET}.$(ClusterId).$(ProcId).err
log							= train_MHc-{MHc}_MA-{MA}_vs_{TARGET}.$(ClusterId).$(ProcId).log
should_transfer_files		= Yes
when_to_transfer_output		= ON_EXIT
request_GPUs				= 1
request_CPUs				= 6
queue""")

		with open(f"train_MHc-{MHc}_MA-{MA}_vs_{TARGET}.sh", "w") as f:
			f.write(f"""#!/bin/sh
source /u/user/choij/scratch/miniconda3/bin/activate
conda activate torch

cd /u/user/choij/scratch/HcToWA/SignalRegion_ver5
python train_parallel.py --mass MHc{MHc}_MA{MA} --channel {CHANNEL} --target {TARGET}
""")
		os.system(f"condor_submit train_MHc-{MHc}_MA-{MA}_vs_{TARGET}.sub")
