import subprocess, psutil
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--channel", "-c", default=None, required=True, type=str, help='channel')
args = parser.parse_args()

#MassPoints = {70: [15, 40, 65],
MassPoints = {70: [40, 65],
              100: [15, 25, 60, 95],
              130: [15, 45, 55, 90, 125],
              160: [15, 45, 75, 85, 120, 155]}


for MHc in MassPoints.keys():
    for MA in MassPoints[MHc]:
        #samples = ['DATA', 'DY', 'ZG', 'ttX', 
        #           'rare', 'VV', 
        #           f'TTToHcToWA_AToMuMu_MHc{MHc}_MA{MA}']
        print(f"MHc: {MHc}, MA: {MA}")
        samples = ['ttX']
        processes = dict()
        pid_list = list()
        for sample in samples:
            command = f"/root/anaconda3/envs/torch/bin/python Analyzer.py -s {sample} -c {args.channel} -m MHc{MHc}_MA{MA}"
            proc = subprocess.Popen(command, shell=True)
            processes[sample] = proc
            pid_list.append(proc.pid)
            
        for sample in samples:
            proc = processes[sample]
            out, err = proc.communicate()
            print(out, err)
            time.sleep(5)

        #while(True):
        #    is_running = False
            # check whether any process is still running
        #    print(psutil.pids())
        #    for pid in proc_list:
        #        if psutil.pid_exists(pid):
        #            is_running = True
        #            print(pid)
        #            break

        #    if not is_running:
        #        print(f"End process for MHc{MHc}_MA{MA}")
        #        break
        #    time.sleep(5)
        exit()
