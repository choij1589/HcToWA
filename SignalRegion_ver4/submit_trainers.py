import os, psutil
import argparse
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument("--channel", "-c", default=None, required=True, type=str, help='channel')
args = parser.parse_args()

MassPoints = {70: [15, 40, 65],
              100: [15, 25, 60, 95],
              130: [15, 45, 55, 90, 125],
              160: [15, 45, 75, 85, 120, 155]}

PID = os.getpid()

for MHc in MassPoints.keys():
    for MA in MassPoints[MHc]:
        print(f"MHc: {MHc}, MA: {MA}")

        os.system(f"python train.py --channel {args.channel} --mass MHc{MHc}_MA{MA} &")

    while True:
        is_running = False
        for p in psutil.process_iter():
            if p.pid == PID:
                continue
            if "python" in p.name():
                is_running = True
                break

        if not is_running:
            print(f"End process for MHc-{MHc}/MA-{MA}")
            break
        sleep(5)
