import os

def do_fit(path, object):
    command = f"""root -q -b 'tempFit.cc("{path}", "{object}")' >> {path.replace("/", "_")}.log"""
    print(command)
    os.system(command)

def get_coef(path, sample):
    with open(f"""{path.replace("/", "_")}.log""", "r") as f:
        while True:
            line = f.readline()
            if (not "COVARIANCE MATRIX CALCULATED SUCCESSFULLY" in line) or ("MATRIX FORCED POS-DEF" in line):
                continue
            else:
                break
        while True:
            line = f.readline()
            if f" coef_{sample} " in line:
                words = line.split(" ")
                words = list(filter(lambda a: a != "", words))
                return float(words[2])
            else:
                continue

def get_coefs(path, object):
    do_fit(path, object)
    if object == "electron":
        samples = ["QCD_EMEnriched", "QCD_bcToE", "W", "DY", "TT", "ST", "VV"]
    elif object == "muon":
        samples = ["QCD_MuEnriched", "W", "DY", "TT", "ST", "VV"]
    else:
        pass
        
    coefs = dict()
    for sample in samples:
        coefs[sample] = get_coef(path, sample)

    # for key, value in coefs.items():
    #     print(f"{key}: {value}")
    os.system(f"""rm {path.replace("/", "_")}.log""")
    return coefs

if __name__ == "__main__":
    coefs = get_coefs("loose/passEle8Path/Central/eta0to0p8", "electron")
