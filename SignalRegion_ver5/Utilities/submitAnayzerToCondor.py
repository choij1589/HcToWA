import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--channel", "-c", default=None, required=True, type=str, help='channel')
parser.add_argument("--sample", "-s", default=None, required=True, type=str, help='target')
args = parser.parse_args()
CHANNEL = args.channel
SAMPLE = args.sample

os.chdir(f"condor_base/analyze/{CHANNEL}")

with open(f"analyze_{SAMPLE}.sub", "w") as f:
	f.write(f"""executable				= analyze_{SAMPLE}.sh
arguments					= $(ClusterId)$(ProcId)
output						= analyze_{SAMPLE}.$(ClusterId).$(ProcId).out
error						= analyze_{SAMPLE}.$(ClusterId).$(ProcId).err
log							= analyze_{SAMPLE}.$(ClusterId).$(ProcId).log
should_transfer_files		= Yes
when_to_transfer_output		= ON_EXIT
request_CPUs				= 1
queue""")

with open(f"analyze_{SAMPLE}.sh", "w") as f:
	f.write(f"""#!/bin/sh
source /u/user/choij/scratch/miniconda3/bin/activate
conda activate torch

cd /u/user/choij/scratch/HcToWA/SignalRegion_ver5
python Analyzer.py --channel {CHANNEL} --sample {SAMPLE}
""")
os.system(f"condor_submit analyze_{SAMPLE}.sub")
