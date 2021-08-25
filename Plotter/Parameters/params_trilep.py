cvs_params = {
        "logy": False,
        "grid": False
}
info_params = {
        "info": "L_{int} = 41.9 fb^{-1}",
        "cms_text": "CMS",
        "extra_text": "Work in progress"
}

muon_params = {
        "muons/1/pt": {
            "x_title": "p_{T}(#mu1)",
            "x_range": [0., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "muons/2/pt": {
            "x_title": "p_{T}(#mu2)",
            "x_range": [0., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "muons/3/pt": {
            "x_title": "p_{T}(#mu3)",
            "x_range": [0., 100.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "muons/1/eta": {
            "x_title": "#eta(#mu1)",
            "x_range": [-2.4, 2.4],
            "y_title": "Events",
            "rebin": 2,
            "ratio_range": [0., 2.0]
        },
        "muons/2/eta": {
            "x_title": "#eta(#mu2)",
            "x_range": [-2.4, 2.4],
            "y_title": "Events",
            "rebin": 2,
            "ratio_range": [0., 2.0]
        },
        "muons/3/eta": {
            "x_title": "#eta(#mu3)",
            "x_range": [-2.4, 2.4],
            "y_title": "Events",
            "rebin": 2,
            "ratio_range": [0., 2.0]
        },
        "muons/1/phi": {
            "x_title": "#phi(#mu1)",
            "x_range": [-3.2, 3.2],
            "y_title": "Events",
            "rebin": 4,
            "ratio_range": [0., 2.0]
        },
        "muons/2/phi": {
            "x_title": "#phi(#mu2)",
            "x_range": [-3.2, 3.2],
            "y_title": "Events",
            "rebin": 4,
            "ratio_range": [0., 2.0]
        },
        "muons/3/phi": {
            "x_title": "#phi(#mu3)",
            "x_range": [-3.2, 3.2],
            "y_title": "Events",
            "rebin": 4,
            "ratio_range": [0., 2.0]
        }
}
electron_params = {
        "electrons/1/pt": {
            "x_title": "p_{T}(e)",
            "x_range": [0., 150.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "electrons/1/eta": {
            "x_title": "#eta(e)",
            "x_range": [-2.4, 2.4],
            "y_title": "Events",
            "rebin": 2,
            "ratio_range": [0., 2.0]
        },
        "electrons/1/phi": {
            "x_title": "#phi(e)",
            "x_range": [-3.2, 3.2],
            "y_title": "Events",
            "rebin": 4,
            "ratio_range": [0., 2.0]
        }
}
jet_params = {
        "jets/1/pt": {
            "x_title": "p_{T}(j1)",
            "x_range": [0., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "jets/2/pt": {
            "x_title": "p_{T}(j2)",
            "x_range": [0., 150.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "jets/3/pt": {
            "x_title": "p_{T}(j3)",
            "x_range": [0., 100.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "jets/1/eta": {
            "x_title": "#eta(j1)",
            "x_range": [-2.4, 2.4],
            "y_title": "Events",
            "rebin": 2,
            "ratio_range": [0., 2.0]
        },
        "jets/2/eta": {
            "x_title": "#eta(j2)",
            "x_range": [-2.4, 2.4],
            "y_title": "Events",
            "rebin": 2,
            "ratio_range": [0., 2.0]
        },
        "jets/3/eta": {
            "x_title": "#eta(j3)",
            "x_range": [-2.4, 2.4],
            "y_title": "Events",
            "rebin": 2,
            "ratio_range": [0., 2.0]
        },
        "jets/1/phi": {
            "x_title": "#phi(j1)",
            "x_range": [-3.2, 3.2],
            "y_title": "Events",
            "rebin": 4,
            "ratio_range": [0., 2.0]
        },
        "jets/2/phi": {
            "x_title": "#phi(j2)",
            "x_range": [-3.2, 3.2],
            "y_title": "Events",
            "rebin": 4,
            "ratio_range": [0., 2.0]
        },
        "jets/3/phi": {
            "x_title": "#phi(j3)",
            "x_range": [-3.2, 3.2],
            "y_title": "Events",
            "rebin": 4,
            "ratio_range": [0., 2.0]
        },
        "jets/size": {
            "x_title": "N_{j}",
            "x_range": [0., 10.],
            "y_title": "Events",
            "ratio_range": [0., 2.0]
        }
}

dimuon_params = {
        "DiMuon/pt": {
            "x_title": "p_{T}(Z)",
            "x_range": [0., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "DiMuon/eta": {
            "x_title": "#eta(Z)",
            "x_range": [-3., 3.],
            "y_title": "Events",
            "rebin": 4,
            "ratio_range": [0., 2.0]
        },
        "DiMuon/phi": {
            "x_title": "#phi(Z)",
            "x_range": [-3.2, 3.2],
            "y_title": "Events",
            "rebin": 4,
            "ratio_range": [0., 2.0]
        },
        "DiMuon/mass": {
            "x_title": "M(Z)",
            "x_range": [80., 103.],
            "y_title": "Events",
            "ratio_range": [0., 2.0]
        }
}

METv_params = {
        "METv/pt": {
            "x_title": "E_{T}^{miss}",
            "x_range": [0., 300.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "METv/phi": {
            "x_title": "#phi(E_{T}^{miss})",
            "x_range": [-3.2, 3.2],
            "y_title": "Events",
            "rebin": 4,
            "ratio_range": [0., 2.0]
        }
}

input_params = {
        # parameters for ML inputs
        "mu1_px": {
            "x_title": "p_{x}(#mu1)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "mu1_py": {
            "x_title": "p_{y}(#mu1)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "mu1_pz": {
            "x_title": "p_{z}(#mu1)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "mu1_mass": {
            "x_title": "M(#mu1)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "mu2_px": {
            "x_title": "p_{x}(#mu2)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "mu2_py": {
            "x_title": "p_{y}(#mu2)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "mu2_pz": {
            "x_title": "p_{z}(#mu2)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "mu2_mass": {
            "x_title": "M(#mu2)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "ele_px": {
            "x_title": "p_{x}(e)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "ele_py": {
            "x_title": "p_{y}(e)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "ele_pz": {
            "x_title": "p_{z}(e)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "ele_mass": {
            "x_title": "M(e)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "j1_px": {
            "x_title": "p_{x}(j1)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "j1_py": {
            "x_title": "p_{y}(j1)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "j1_pz": {
            "x_title": "p_{z}(j1)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "j1_mass": {
            "x_title": "M(j1)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "j1_bscore": {
            "x_title": "b-score(j1)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "j2_px": {
            "x_title": "p_{x}(j2)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "j2_py": {
            "x_title": "p_{y}(j2)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "j2_pz": {
            "x_title": "p_{z}(j2)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "j2_mass": {
            "x_title": "M(j2)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "j2_bscore": {
            "x_title": "b-score(j2)",
            "x_range": [-200., 200.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "dR_mu1mu2": {
            "x_title": "#Delta R(#mu1 #mu2)",
            "x_range": [0., 10.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "dR_mu1ele": {
            "x_title": "#Delta R(#mu1 e)",
            "x_range": [0., 10.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "dR_mu2ele": {
            "x_title": "#Delta R(#mu2 ele)",
            "x_range": [0., 10.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "dR_j1ele": {
            "x_title": "#Delta R(j1 e)",
            "x_range": [0., 10.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "dR_j2ele": {
            "x_title": "#Delta R(j2 e)",
            "x_range": [0., 10.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "dR_j1j2": {
            "x_title": "#Delta R(j1 j2)",
            "x_range": [0., 10.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "HT": {
            "x_title": "H_{T}",
            "x_range": [0., 600.],
            "y_title": "Events",
            "rebin": 10,
            "ratio_range": [0., 2.0]
        },
        "MT": {
            "x_title": "M_{T}",
            "x_range": [0., 300.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "LT": {
            "x_title": "L_{T}",
            "x_range": [0., 300.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "MET": {
            "x_title": "E_{T}^{miss}",
            "x_range": [0., 300.],
            "y_title": "Events",
            "rebin": 5,
            "ratio_range": [0., 2.0]
        },
        "Nj": {
            "x_title": "N_{j}",
            "x_range": [0., 10.],
            "y_title": "Events",
            "ratio_range": [0., 2.0]
        },
        "Nb": {
            "x_title": "N_{b}",
            "x_range": [0., 10.],
            "y_title": "Events",
            "ratio_range": [0., 2.0]
        },
        "avg_dRjets": {
            "x_title": "<#Delta R(jets):>",
            "x_range": [0., 10.],
            "y_title": "Events",
            "ratio_range": [0., 2.0]
        },
        "avg_bscore": {
            "x_title": "<bscore>",
            "x_range": [0., 1.],
            "y_title": "Events",
            "rebin": 2,
            "ratio_range": [0., 2.0]
        }
}

score_params = {
        "score_fake": {
            "x_title": "score(fake)",
            "x_range": [0., 1.],
            "y_title": "Events",
            "ratio_range": [0., 2.0]
        },
        "score_ttX": {
            "x_title": "score(ttX)",
            "x_range": [0., 1.],
            "y_title": "Events",
            "ratio_range": [0., 2.0]
        }
}














