cvs_params = {
    "logy": False,
    "grid": False
}
info_params = {
    "info": "L_{int} = 41.9 fb^{-1}",
    "cms_text": "CMS",
    "extra_text": "Work in progress"
}
param_list = dict()
param_list["electrons/1/pt"] = {
    "x_title": "p_{T}(e) (GeV)",
    "x_range": [0., 250.],
    "y_title": "Events",
    "rebin": 10,
    "ratio_range": [0., 2.0]
}
param_list["electrons/1/eta"] = {
    "x_title": "#eta(e)",
    "x_range": [-2.5, 2.5],
    "y_title": "Events",
    "rebin": 2,
    "ratio_range": [0., 2.0]
}
param_list["electrons/1/phi"] = {
    "x_title": "#phi(e)",
    "x_range": [-3.2, 3.2],
    "y_title": "Events",
    "rebin": 2,
    "ratio_range": [0., 2.0]
}
param_list["muons/1/pt"] = {
    "x_title": "p_{T}(#mu1) (GeV)",
    "x_range": [0., 250.],
    "y_title": "Events",
    "rebin": 10,
    "ratio_range": [0., 2.0]
}
param_list["muons/2/pt"] = {
    "x_title": "p_{T}(#mu2) (GeV)",
    "x_range": [0., 200.],
    "y_title": "Events",
    "rebin": 10,
    "ratio_range": [0., 2.0]
}
param_list["muons/3/pt"] = {
    "x_title": "p_{T}(#mu3) (GeV)",
    "x_range": [0., 150.],
    "y_title": "Events",
    "rebin": 5,
    "ratio_range": [0., 2.0]
}
param_list["muons/1/eta"] = {
    "x_title": "#eta(#mu1)",
    "x_range": [-2.4, 2.4],
    "y_title": "Events",
    "rebin": 2,
    "ratio_range": [0., 2.0]
}
param_list["muons/2/eta"] = {
    "x_title": "#eta(#mu2)",
    "x_range": [-2.4, 2.4],
    "y_title": "Events",
    "rebin": 2,
    "ratio_range": [0., 2.0]
}
param_list["muons/3/eta"] = {
    "x_title": "#eta(#mu3)",
    "x_range": [-2.4, 2.4],
    "y_title": "Events",
    "rebin": 2,
    "ratio_range": [0., 2.0]
}
param_list["muons/1/phi"] = {
    "x_title": "#phi(#mu1)",
    "x_range": [-3.2, 3.2],
    "y_title": "Events",
    "rebin": 2,
    "ratio_range": [0., 2.0]
}
param_list["muons/2/phi"] = {
    "x_title": "#phi(#mu2)",
    "x_range": [-3.2, 3.2],
    "y_title": "Events",
    "rebin": 2,
    "ratio_range": [0., 2.0]
}
param_list["muons/3/phi"] = {
    "x_title": "#phi(#mu3)",
    "x_range": [-3.2, 3.2],
    "y_title": "Events",
    "rebin": 2,
    "ratio_range": [0., 2.0]
}
param_list["jets/1/pt"] = {
    "x_title": "p_{T}(j1) (GeV)",
    "x_range": [0., 250.],
    "y_title": "Events",
    "rebin": 10,
    "ratio_range": [0., 2.0]
}
param_list["jets/2/pt"] = {
    "x_title": "p_{T}(j2) (GeV)",
    "x_range": [0., 200.],
    "y_title": "Events",
    "rebin": 10,
    "ratio_range": [0., 2.0]
}
param_list["jets/1/eta"] = {
    "x_title": "#eta(j1)",
    "x_range": [-2.4, 2.4],
    "y_title": "Events",
    "rebin": 2,
    "ratio_range": [0., 2.0]
}
param_list["jets/2/eta"] = {
    "x_title": "#eta(j2)",
    "x_range": [-2.4, 2.4],
    "y_title": "Events",
    "rebin": 2,
    "ratio_range": [0., 2.0]
}
param_list["jets/1/phi"] = {
    "x_title": "#phi(j1)",
    "x_range": [-3.2, 3.2],
    "y_title": "Events",
    "rebin": 2,
    "ratio_range": [0., 2.0]
}
param_list["jets/2/phi"] = {
    "x_title": "#phi(j2)",
    "x_range": [-3.2, 3.2],
    "y_title": "Events",
    "rebin": 2,
    "ratio_range": [0., 2.0]
}
param_list["jets/Nj"] = {
    "x_title": "N_{j}",
    "x_range": [0., 10.],
    "y_title": "Events",
    "ratio_range": [0., 2.0]
}
param_list["ZMass"] = {
    "x_title": "M(Z)",
    "x_range": [80., 103.],
    "y_title": "Events",
    "ratio_range": [0., 2.0]
}
param_list["MET"] = {
    "x_title": "E_{T}^{miss}",
    "x_range": [0., 300.],
    "y_title": "Events",
    "rebin": 10,
    "ratio_range": [0., 2.0]
}
param_list["nPV"] = {
    "x_title": "nPV",
    "x_range": [0., 100.],
    "y_title": "Events",
    "ratio_range": [0., 2.0]
}
param_list["nPileUp"] = {
    "x_title": "nPileUp",
    "x_range": [0., 100.],
    "y_title": "Events",
    "ratio_range": [0., 2.0]
}
