from BinnedAndIncl import BinnedAndIncl
from Parameters.binned_and_incl_params import params
from ROOT import TFile

def plot_maker(obs, root_files, weights, selector_output, output_path, store_name):
    cvs_params = params[obs]["cvs_params"]
    hist_params = params[obs]['hist_params']
    info_params = params[obs]["info_params"]

    hist_name = obs
    hist_incl = None
    hists_binned = {}
    for file_name in file_names:
        dir_name = "Unknown"

        # combine ee and mm channels
        scale = weights[file_name]*lumi*1000
        hist_path = dir_name + "/" + hist_name
        #print(hist_path + selector_arg + "_ee")
        hist_ee = root_files[file_name].Get(
            hist_path + selector_arg + "_ee")
        hist_mm = root_files[file_name].Get(
            hist_path + selector_arg + "_mm")
        hist = hist_ee.Clone(file_name + selector_arg + "_clone")
        hist.Add(hist_mm)
        hist.Scale(scale)

        key = file_name
        if selector_arg == "":
            file_name += "_dressedLep"
        elif selector_arg == "_prefsr":
            file_name += "_matchedGenJet"
        else:
            file_name += selector_arg

        if "012j" in file_name:
            hist_incl = hist
        else:
            hists_binned[key] = hist

    plotter = BinnedAndIncl(cvs_params, hist_params, info_params)
    plotter.get_hists(hist_incl, hists_binned)
    plotter.combine()
    if selector_arg == "":
        path = output_path + store_name + "_" + obs + "_dressedLep" + ".png"
    elif selector_arg == "_prefsr":
        path = output_path + store_name + "_" + obs + "_matchedGenJet" + ".png"
    else:
        path = output_path + store_name + "_" + obs + selector_arg + ".png"
    plotter.save(path)

# initial settings
file_names = ["DYm50_012j_nlo_ewparams_cp5", "DYm50_0j_nlo_ewparams_fxfxon_cp5",
            	"DYm50_1j_nlo_ewparams_cp5", "DYm50_2j_nlo_ewparams_cp5"]
weights = {
    "DYm50_012j_nlo_ewparams_cp5": 2.80340430E-03,
    "DYm50_0j_nlo_ewparams_fxfxon_cp5": 9.69581073E-04,
    "DYm50_1j_nlo_ewparams_cp5": 7.68607325E-04,
    "DYm50_2j_nlo_ewparams_cp5": 4.40995922E-04,
}
observables = ["ZMass", "yZ", "ptZ", "phiZ",
                "ptl1", "ptl2", "etal1", "etal2", "phil1", "phil2", "nLeptons",
                "ptj1", "ptj2", "etaj1", "etaj2", "phij1", "phij2", "nJets"]
zboson = ['ZMass', 'yZ', 'ptZ', 'phiZ']
leptons = ['ptl1', 'ptl2', 'etal1', 'etal2', 'phil1', 'phil2', 'nLeptons']
jets = ['ptj1', 'ptj2', 'etaj1', 'etaj2', 'phij1', 'phij2', 'nJets']
lumi = 150.  # fb^-1
selector_arg = "_prefsr"
output_path = "/root/workspace/GenValidation/www/VGenStudies/DYm50_nlo_ewparams_cp5/"
# get histograms
root_files = {}
selector_output = "/root/workspace/GenValidation/SelectorOutput/DrellYan/"
store_name = "DYm50_nlo_ewparams_cp5"
for name in file_names:
    this_path = selector_output + name + ".root"
    root_files[name] = TFile(this_path)

for obs in observables:
    plot_maker(obs, root_files, weights, selector_output, output_path, store_name)
