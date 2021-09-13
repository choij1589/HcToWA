import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--channel", "-c", default=None, required=True, type=str, help='channel')
parser.add_argument("--sample", "-s", default=None, required=True, type=str, help='target')
parser.add_argument("--mass", "-m", default=None, required=True, type=str, help="signal mass point")
args = parser.parse_args()
CHANNEL = args.channel
SAMPLE = args.sample
MASS_POINT = args.mass

path = f"condor_base/analyzer/{CHANNEL}/{MASS_POINT}"
if not os.path.exists(path):
	os.makedirs(path)
os.chdir(path)

with open(f"analyze_{SAMPLE}.sub", "w") as f:
	f.write(f"""executable				= analyze_{SAMPLE}.sh
arguments					= $(ClusterId)$(ProcId)
output						= analyze_{SAMPLE}.$(ClusterId).$(ProcId).out
error						= analyze_{SAMPLE}.$(ClusterId).$(ProcId).err
log							= analyze_{SAMPLE}.$(ClusterId).$(ProcId).log
should_transfer_files		= Yes
when_to_transfer_output		= ON_EXIT
request_CPUs				= 1
request_memory				= 4000
queue""")

with open(f"analyze_{SAMPLE}.sh", "w") as f:
	f.write(f"""#!/bin/sh
source /u/user/choij/scratch/miniconda3/bin/activate
conda activate torch

cd /u/user/choij/scratch/HcToWA/SignalRegion_ver5
python Analyzer.py --channel {CHANNEL} --sample {SAMPLE} --mass {MASS_POINT}
""")
os.system(f"condor_submit analyze_{SAMPLE}.sub")
