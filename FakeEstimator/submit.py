import os, shutil
import argparse
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("--object", "-o", default=None, required=True, type=str, help="electron or muon")
parser.add_argument("--sample", "-s", default=None, required=True, type=str, help="sample to submit")
parser.add_argument("--num_core", "-n", default=None, required=True, type=int, help="number of cores to submit")
args = parser.parse_args()

def condor_jdl():
    jdl = f"""Universe             = vanilla
Executable           = fakerate_{args.object}_{args.sample}.sh
GetEnv               = false

WhenToTransferOutput = On_Exit_Or_Evict
ShouldTransferFiles  = yes
want_graceful_removal = true
request_memory       = 2000
request_disk         = 2048000
use_x509userproxy = False

# stop jobs from running if they blow up in size or memory
periodic_hold        = (DiskUsage/1024 > 10.0*2000 ||  ImageSize/1024 > RequestMemory*2)

+JobFlavour = "workday"
arguments             = $(Process)
output                = /data6/Users/choij/CMSSW/CMSSW_11_3_0_pre2/src/HcToWA/FakeEstimator/Outputs/{args.object}/{args.sample}/output/job_$(Process).out
error                 = /data6/Users/choij/CMSSW/CMSSW_11_3_0_pre2/src/HcToWA/FakeEstimator/Outputs/{args.object}/{args.sample}/error/job_$(Process).err
log                   = /data6/Users/choij/CMSSW/CMSSW_11_3_0_pre2/src/HcToWA/FakeEstimator/Outputs/{args.object}/{args.sample}/log/job_$(Process).log

queue {args.num_core}"""

    return jdl


def run():
    script = f"""#!/bin/sh
cd /data6/Users/choij/CMSSW/CMSSW_11_3_0_pre2/src/HcToWA/FakeEstimator
process=$1

source /cvmfs/cms.cern.ch/cmsset_default.sh
eval `scramv1 runtime -sh`
python3 Analyzer.py -s {args.sample} -o {args.object} -f syst -n {args.num_core} -t ${{process}}"""

    return script

if __name__ == "__main__":
    # manage directories
    base_dir = f"Outputs/{args.object}/{args.sample}"
    try:
        if os.listdir(base_dir):
            print(f"Warning: Deleting {base_dir}...")
            shutil.rmtree(base_dir)
        else:
            pass
    except:
        pass
    os.mkdir(base_dir)
    os.mkdir(f"{base_dir}/output")
    os.mkdir(f"{base_dir}/error")
    os.mkdir(f"{base_dir}/log")
    os.mkdir(f"{base_dir}/scripts")

    # make scripts
    with open(f"{base_dir}/scripts/condor.jdl", "w") as f:
        f.write(condor_jdl())
    with open(f"{base_dir}/scripts/fakerate_{args.object}_{args.sample}.sh", "w") as f:
        f.write(run())

    os.chdir(f"{base_dir}/scripts")
    os.system("condor_submit condor.jdl")

	
    
