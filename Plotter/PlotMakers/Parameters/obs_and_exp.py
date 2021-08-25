cvs_params = {
    "logy": True,
    "grid": True
}
info_params = {
    "info": "L^{int} = 41.5 fb^{-1}",
    "cms_text": "CMS",
    "extra_text": "Work on progress"
}

param_set = {
    "DY_dimu": {
        "ptmu1": {
            "path": "/muons_tight/1/pt",
            "hist_params": {
                "x_title": "p_{T}(#mu1)",
                "x_range": [0., 200.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptmu2": {
            "path": "/muons_tight/2/pt",
            "hist_params": {
                "x_title": "p_{T}(#mu2)",
                "x_range": [0., 200.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etamu1": {
            "path": "/muons_tight/1/eta",
            "hist_params": {
                "x_title": "#eta(#mu1)",
                "x_range": [-2.4, 2.4],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etamu2": {
            "path": "/muons_tight/2/eta",
            "hist_params": {
                "x_title": "#eta(#mu2)",
                "x_range": [-2.4, 2.4],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phimu1": {
            "path": "/muons_tight/1/phi",
            "hist_params": {
                "x_title": "#phi(#mu1)",
                "x_range": [-3.2, 3.2],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phimu2": {
            "path": "/muons_tight/2/phi",
            "hist_params": {
                "x_title": "#phi(#mu2)",
                "x_range": [-3.2, 3.2],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptj1": {
            "path": "/jets_cleaned/1/pt",
            "hist_params": {
                "x_title": "p_{T}(j1)",
                "x_range": [0., 200.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptj2": {
            "path": "/jets_cleaned/2/pt",
            "hist_params": {
                "x_title": "p_{T}(j2)",
                "x_range": [0., 200.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptj3": {
            "path": "/jets_cleaned/3/pt",
            "hist_params": {
                "x_title": "p_{T}(j3)",
                "x_range": [0., 200.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etaj1": {
            "path": "/jets_cleaned/1/eta",
            "hist_params": {
                "x_title": "#eta(j1)",
                "x_range": [-2.4, 2.4],
                "rebin": 3,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etaj2": {
            "path": "/jets_cleaned/2/eta",
            "hist_params": {
                "x_title": "#eta(j2)",
                "x_range": [-2.4, 2.4],
                "rebin": 3,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etaj3": {
            "path": "/jets_cleaned/3/eta",
            "hist_params": {
                "x_title": "#eta(j3)",
                "x_range": [-2.4, 2.4],
                "rebin": 3,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phij1": {
            "path": "/jets_cleaned/1/phi",
            "hist_params": {
                "x_title": "#phi(j1)",
                "x_range": [-3.2, 3.2],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phij2": {
            "path": "/jets_cleaned/2/phi",
            "hist_params": {
                "x_title": "#phi(j2)",
                "x_range": [-3.2, 3.2],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phij3": {
            "path": "/jets_cleaned/3/phi",
            "hist_params": {
                "x_title": "#phi(j3)",
                "x_range": [-3.2, 3.2],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "nJets": {
            "path": "/jets_cleaned/size",
            "hist_params": {
                "x_title": "N_{j}",
                "x_range": [0, 10],
                "y_title": "Events",
                "ratio_range": [0., 2.]
            }
        },
        "METv": {
            "path": "/METv/pt",
            "hist_params": {
                "x_title": "E_{T}^{miss}",
                "x_range": [0., 150.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0., 2.0]
            }
        },
        "etaMETv": {
            "path": "/METv/eta",
            "hist_params": {
                "x_title": "#eta(E_{T}^{miss})",
                "y_title": "Events",
                "ratio_range": [0., 2.0]
            }
        },
        "phiMETv": {
            "path": "/METv/phi",
            "hist_params": {
                "x_title": "#phi(E_{T}^{miss})",
                "y_title": "Events",
                "rebin": 2,
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptZ": {
            "path": "/ZCand/pt",
            "hist_params": {
                "x_title": "p_T(Z)",
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etaZ": {
            "path": "/ZCand/eta",
            "hist_params": {
                "x_title": "#eta(Z)",
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phiZ": {
            "path": "/ZCand/phi",
            "hist_params": {
                "x_title": "#phi(Z)",
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "Zmass": {
            "path": "/ZCand/mass",
            "hist_params": {
                "x_title": "M(Z)",
                "x_range": [75., 108.],
                "y_title": "Events / GeV",
                "ratio_range": [0.5, 1.5]
            }
        }
    },
    "TT_dimu": {
        "ptmu1": {
            "path": "/muons_tight/1/pt",
            "hist_params": {
                "x_title": "p_{T}(#mu1)",
                "x_range": [0., 200.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptmu2": {
            "path": "/muons_tight/2/pt",
            "hist_params": {
                "x_title": "p_{T}(#mu2)",
                "x_range": [0., 200.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etamu1": {
            "path": "/muons_tight/1/eta",
            "hist_params": {
                "x_title": "#eta(#mu1)",
                "x_range": [-2.4, 2.4],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etamu2": {
            "path": "/muons_tight/2/eta",
            "hist_params": {
                "x_title": "#eta(#mu2)",
                "x_range": [-2.4, 2.4],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phimu1": {
            "path": "/muons_tight/1/phi",
            "hist_params": {
                "x_title": "#phi(#mu1)",
                "x_range": [-3.2, 3.2],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phimu2": {
            "path": "/muons_tight/2/phi",
            "hist_params": {
                "x_title": "#phi(#mu2)",
                "x_range": [-3.2, 3.2],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptj1": {
            "path": "/jets_cleaned/1/pt",
            "hist_params": {
                "x_title": "p_{T}(j1)",
                "x_range": [0., 200.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptj2": {
            "path": "/jets_cleaned/2/pt",
            "hist_params": {
                "x_title": "p_{T}(j2)",
                "x_range": [0., 200.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptj3": {
            "path": "/jets_cleaned/3/pt",
            "hist_params": {
                "x_title": "p_{T}(j3)",
                "x_range": [0., 200.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etaj1": {
            "path": "/jets_cleaned/1/eta",
            "hist_params": {
                "x_title": "#eta(j1)",
                "x_range": [-2.4, 2.4],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etaj2": {
            "path": "/jets_cleaned/2/eta",
            "hist_params": {
                "x_title": "#eta(j2)",
                "x_range": [-2.4, 2.4],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etaj3": {
            "path": "/jets_cleaned/3/eta",
            "hist_params": {
                "x_title": "#eta(j3)",
                "x_range": [-2.4, 2.4],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phij1": {
            "path": "/jets_cleaned/1/phi",
            "hist_params": {
                "x_title": "#phi(j1)",
                "x_range": [-3.2, 3.2],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phij2": {
            "path": "/jets_cleaned/2/phi",
            "hist_params": {
                "x_title": "#phi(j2)",
                "x_range": [-3.2, 3.2],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phij3": {
            "path": "/jets_cleaned/3/phi",
            "hist_params": {
                "x_title": "#phi(j3)",
                "x_range": [-3.2, 3.2],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "nJets": {
            "path": "/jets_cleaned/size",
            "hist_params": {
                "x_title": "N_{j}",
                "x_range": [0, 10],
                "y_title": "Events",
                "ratio_range": [0., 2.]
            }
        },
        "ptb1": {
            "path": "/bjets_cleaned/1/pt",
            "hist_params": {
                "x_title": "p_{T}(b1)",
                "x_range": [0., 200.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptb2": {
            "path": "/bjets_cleaned/2/pt",
            "hist_params": {
                "x_title": "p_{T}(b2)",
                "x_range": [0., 200.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptb3": {
            "path": "/bjets_cleaned/3/pt",
            "hist_params": {
                "x_title": "p_{T}(b3)",
                "x_range": [0., 200.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etab1": {
            "path": "/bjets_cleaned/1/eta",
            "hist_params": {
                "x_title": "#eta(b1)",
                "x_range": [-2.4, 2.4],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etab2": {
            "path": "/bjets_cleaned/2/eta",
            "hist_params": {
                "x_title": "#eta(b2)",
                "x_range": [-2.4, 2.4],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etab3": {
            "path": "/bjets_cleaned/3/eta",
            "hist_params": {
                "x_title": "#eta(b3)",
                "x_range": [-2.4, 2.4],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phib1": {
            "path": "/bjets_cleaned/1/phi",
            "hist_params": {
                "x_title": "#phi(b1)",
                "x_range": [-3.2, 3.2],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phib2": {
            "path": "/bjets_cleaned/2/phi",
            "hist_params": {
                "x_title": "#phi(b2)",
                "x_range": [-3.2, 3.2],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phib3": {
            "path": "/bjets_cleaned/3/phi",
            "hist_params": {
                "x_title": "#phi(b3)",
                "x_range": [-3.2, 3.2],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "nBJets": {
            "path": "/bjets_cleaned/size",
            "hist_params": {
                "x_title": "N_{b}",
                "x_range": [0, 10],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "METv": {
            "path": "/METv/pt",
            "hist_params": {
                "x_title": "E_{T}^{miss}",
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0., 2.0]
            }
        },
        "etaMETv": {
            "path": "/METv/eta",
            "hist_params": {
                "x_title": "#eta(E_{T}^{miss})",
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0., 2.0]
            }
        },
        "phiMETv": {
            "path": "/METv/phi",
            "hist_params": {
                "x_title": "#phi(E_{T}^{miss})",
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptZ": {
            "path": "/ZCand/pt",
            "hist_params": {
                "x_title": "p_T(Z)",
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etaZ": {
            "path": "/ZCand/eta",
            "hist_params": {
                "x_title": "#eta(Z)",
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phiZ": {
            "path": "/ZCand/phi",
            "hist_params": {
                "x_title": "#phi(Z)",
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "Zmass": {
            "path": "/ZCand/mass",
            "hist_params": {
                "x_title": "M(Z)",
                "x_range": [0., 200.],
                "y_title": "Events / GeV",
                "ratio_range": [0.5, 1.5]
            }
        }
    },
    "TT_emu": {
        "ptmu1": {
            "path": "/muons_tight/1/pt",
            "hist_params": {
                "x_title": "p_{T}(#mu1)",
                "x_range": [0., 200.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "pte1": {
            "path": "/electrons_tight/1/pt",
            "hist_params": {
                "x_title": "p_{T}(e1)",
                "x_range": [0., 200.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etamu1": {
            "path": "/muons_tight/1/eta",
            "hist_params": {
                "x_title": "#eta(#mu1)",
                "x_range": [-2.4, 2.4],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etae1": {
            "path": "/electrons_tight/1/eta",
            "hist_params": {
                "x_title": "#eta(e1)",
                "x_range": [-2.5, 2.5],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phimu1": {
            "path": "/muons_tight/1/phi",
            "hist_params": {
                "x_title": "#phi(#mu1)",
                "x_range": [-3.2, 3.2],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phie1": {
            "path": "/electrons_tight/1/phi",
            "hist_params": {
                "x_title": "#phi(e1)",
                "x_range": [-3.2, 3.2],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptj1": {
            "path": "/jets_cleaned/1/pt",
            "hist_params": {
                "x_title": "p_{T}(j1)",
                "x_range": [0., 200.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptj2": {
            "path": "/jets_cleaned/2/pt",
            "hist_params": {
                "x_title": "p_{T}(j2)",
                "x_range": [0., 200.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptj3": {
            "path": "/jets_cleaned/3/pt",
            "hist_params": {
                "x_title": "p_{T}(j3)",
                "x_range": [0., 200.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etaj1": {
            "path": "/jets_cleaned/1/eta",
            "hist_params": {
                "x_title": "#eta(j1)",
                "x_range": [-2.4, 2.4],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etaj2": {
            "path": "/jets_cleaned/2/eta",
            "hist_params": {
                "x_title": "#eta(j2)",
                "x_range": [-2.4, 2.4],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etaj3": {
            "path": "/jets_cleaned/3/eta",
            "hist_params": {
                "x_title": "#eta(j3)",
                "x_range": [-2.4, 2.4],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phij1": {
            "path": "/jets_cleaned/1/phi",
            "hist_params": {
                "x_title": "#phi(j1)",
                "x_range": [-3.2, 3.2],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phij2": {
            "path": "/jets_cleaned/2/phi",
            "hist_params": {
                "x_title": "#phi(j2)",
                "x_range": [-3.2, 3.2],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phij3": {
            "path": "/jets_cleaned/3/phi",
            "hist_params": {
                "x_title": "#phi(j3)",
                "x_range": [-3.2, 3.2],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "nJets": {
            "path": "/jets_cleaned/size",
            "hist_params": {
                "x_title": "N_{j}",
                "x_range": [0, 10],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptb1": {
            "path": "/bjets_cleaned/1/pt",
            "hist_params": {
                "x_title": "p_{T}(b1)",
                "x_range": [0., 200.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptb2": {
            "path": "/bjets_cleaned/2/pt",
            "hist_params": {
                "x_title": "p_{T}(b2)",
                "x_range": [0., 200.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptb3": {
            "path": "/bjets_cleaned/3/pt",
            "hist_params": {
                "x_title": "p_{T}(b3)",
                "x_range": [0., 200.],
                "rebin": 5,
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etab1": {
            "path": "/bjets_cleaned/1/eta",
            "hist_params": {
                "x_title": "#eta(b1)",
                "x_range": [-2.4, 2.4],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etab2": {
            "path": "/bjets_cleaned/2/eta",
            "hist_params": {
                "x_title": "#eta(b2)",
                "x_range": [-2.4, 2.4],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etab3": {
            "path": "/bjets_cleaned/3/eta",
            "hist_params": {
                "x_title": "#eta(b3)",
                "x_range": [-2.4, 2.4],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phib1": {
            "path": "/bjets_cleaned/1/phi",
            "hist_params": {
                "x_title": "#phi(b1)",
                "x_range": [-3.2, 3.2],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phib2": {
            "path": "/bjets_cleaned/2/phi",
            "hist_params": {
                "x_title": "#phi(b2)",
                "x_range": [-3.2, 3.2],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phib3": {
            "path": "/bjets_cleaned/3/phi",
            "hist_params": {
                "x_title": "#phi(b3)",
                "x_range": [-3.2, 3.2],
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "nBJets": {
            "path": "/bjets_cleaned/size",
            "hist_params": {
                "x_title": "N_{b}",
                "x_range": [0, 10],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "METv": {
            "path": "/METv/pt",
            "hist_params": {
                "x_title": "E_{T}^{miss}",
                "rebin": 5,
                "y_title": "Events/ 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etaMETv": {
            "path": "/METv/eta",
            "hist_params": {
                "x_title": "#eta(E_{T}^{miss})",
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phiMETv": {
            "path": "/METv/phi",
            "hist_params": {
                "x_title": "#phi(E_{T}^{miss})",
                "rebin": 2,
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        }
    },
    "Pre_3mu": {
        "ptmu1": {
            "path": "/muons_tight/1/pt",
            "hist_params": {
                "rebin": 5,
                "x_title": "p_{T}(#mu1)",
                "x_range": [0., 200.],
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptmu2": {
            "path": "/muons_tight/2/pt",
            "hist_params": {
                "rebin": 5,
                "x_title": "p_{T}(#mu2)",
                "x_range": [0., 200.],
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptmu3": {
            "path": "/muons_tight/3/pt",
            "hist_params": {
                "rebin": 5,
                "x_title": "p_{T}(#mu3)",
                "x_range": [0., 150.],
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etamu1": {
            "path": "/muons_tight/1/eta",
            "hist_params": {
                "rebin": 4,
                "x_title": "#eta(#mu1)",
                "x_range": [-2.4, 2.4],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etamu2": {
            "path": "/muons_tight/2/eta",
            "hist_params": {
                "rebin": 4,
                "x_title": "#eta(#mu2)",
                "x_range": [-2.4, 2.4],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etamu3": {
            "path": "/muons_tight/3/eta",
            "hist_params": {
                "rebin": 4,
                "x_title": "#eta(#mu3)",
                "x_range": [-2.4, 2.4],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phimu1": {
            "path": "/muons_tight/1/phi",
            "hist_params": {
                "rebin": 4,
                "x_title": "#phi(#mu1)",
                "x_range": [-3.2, 3.2],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phimu2": {
            "path": "/muons_tight/2/phi",
            "hist_params": {
                "rebin": 4,
                "x_title": "#phi(#mu2)",
                "x_range": [-3.2, 3.2],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phimu3": {
            "path": "/muons_tight/3/phi",
            "hist_params": {
                "rebin": 4,
                "x_title": "#phi(#mu3)",
                "x_range": [-3.2, 3.2],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptj1": {
            "path": "/jets_cleaned/1/pt",
            "hist_params": {
                "rebin": 5,
                "x_title": "p_{T}(j1)",
                "x_range": [0., 200.],
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptj2": {
            "path": "/jets_cleaned/2/pt",
            "hist_params": {
                "rebin": 5,
                "x_title": "p_{T}(j2)",
                "x_range": [0., 200.],
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptj3": {
            "path": "/jets_cleaned/3/pt",
            "hist_params": {
                "rebin": 5,
                "x_title": "p_{T}(j3)",
                "x_range": [0., 200.],
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etaj1": {
            "path": "/jets_cleaned/1/eta",
            "hist_params": {
                "rebin": 6,
                "x_title": "#eta(j1)",
                "x_range": [-2.4, 2.4],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etaj2": {
            "path": "/jets_cleaned/2/eta",
            "hist_params": {
                "rebin": 6,
                "x_title": "#eta(j2)",
                "x_range": [-2.4, 2.4],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etaj3": {
            "path": "/jets_cleaned/3/eta",
            "hist_params": {
                "rebin": 6,
                "x_title": "#eta(j3)",
                "x_range": [-2.4, 2.4],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phij1": {
            "path": "/jets_cleaned/1/phi",
            "hist_params": {
                "rebin": 4,
                "x_title": "#phi(j1)",
                "x_range": [-3.2, 3.2],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phij2": {
            "path": "/jets_cleaned/2/phi",
            "hist_params": {
                "rebin": 4,
                "x_title": "#phi(j2)",
                "x_range": [-3.2, 3.2],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phij3": {
            "path": "/jets_cleaned/3/phi",
            "hist_params": {
                "rebin": 4,
                "x_title": "#phi(j3)",
                "x_range": [-3.2, 3.2],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "nJets": {
            "path": "/jets_cleaned/size",
            "hist_params": {
                "x_title": "N_{j}",
                "x_range": [0, 10],
                "y_title": "Events",
                "ratio_range": [0., 2.]
            }
        },
        "ptb1": {
            "path": "/bjets_cleaned/1/pt",
            "hist_params": {
                "rebin": 5,
                "x_title": "p_{T}(b1)",
                "x_range": [0., 200.],
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptb2": {
            "path": "/bjets_cleaned/2/pt",
            "hist_params": {
                "rebin": 5,
                "x_title": "p_{T}(b2)",
                "x_range": [0., 200.],
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptb3": {
            "path": "/bjets_cleaned/3/pt",
            "hist_params": {
                "rebin": 5,
                "x_title": "p_{T}(b3)",
                "x_range": [0., 200.],
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etab1": {
            "path": "/bjets_cleaned/1/eta",
            "hist_params": {
                "rebin": 6,
                "x_title": "#eta(b1)",
                "x_range": [-2.4, 2.4],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etab2": {
            "path": "/bjets_cleaned/2/eta",
            "hist_params": {
                "rebin": 6,
                "x_title": "#eta(b2)",
                "x_range": [-2.4, 2.4],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etab3": {
            "path": "/bjets_cleaned/3/eta",
            "hist_params": {
                "rebin": 6,
                "x_title": "#eta(b3)",
                "x_range": [-2.4, 2.4],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phib1": {
            "path": "/bjets_cleaned/1/phi",
            "hist_params": {
                "rebin": 4,
                "x_title": "#phi(b1)",
                "x_range": [-3.2, 3.2],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phib2": {
            "path": "/bjets_cleaned/2/phi",
            "hist_params": {
                "rebin": 4,
                "x_title": "#phi(b2)",
                "x_range": [-3.2, 3.2],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phib3": {
            "path": "/bjets_cleaned/3/phi",
            "hist_params": {
                "rebin": 4,
                "x_title": "#phi(b3)",
                "x_range": [-3.2, 3.2],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "nBJets": {
            "path": "/bjets_cleaned/size",
            "hist_params": {
                "x_title": "N_{b}",
                "x_range": [0, 10],
                "y_title": "Events",
                "ratio_range": [0., 2.]
            }
        },
        "METv": {
            "path": "/METv/pt",
            "hist_params": {
                "rebin": 5,
                "x_title": "E_{T}^{miss}",
                "x_range": [0., 200.],
                "y_title": "Events/ 5 GeV",
                "ratio_range": [0., 2.0]
            }
        },
        "etaMETv": {
            "path": "/METv/eta",
            "hist_params": {
                "rebin": 4,
                "x_title": "#eta(E_{T}^{miss})",
                "y_title": "Events",
                "ratio_range": [0., 2.0]
            }
        },
        "phiMETv": {
            "path": "/METv/phi",
            "hist_params": {
                "rebin": 4,
                "x_title": "#phi(E_{T}^{miss})",
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
    },
    "Pre_1e2mu": {
        "ptmu1": {
            "path": "/muons_tight/1/pt",
            "hist_params": {
                "rebin": 5,
                "x_title": "p_{T}(#mu1)",
                "x_range": [0., 200.],
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptmu2": {
            "path": "/muons_tight/2/pt",
            "hist_params": {
                "rebin": 5,
                "x_title": "p_{T}(#mu2)",
                "x_range": [0., 200.],
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "pte1": {
            "path": "/electrons_tight/1/pt",
            "hist_params": {
                "rebin": 5,
                "x_title": "p_{T}(e1)",
                "x_range": [0., 200.],
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etamu1": {
            "path": "/muons_tight/1/eta",
            "hist_params": {
                "rebin": 4,
                "x_title": "#eta(#mu1)",
                "x_range": [-2.4, 2.4],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etamu2": {
            "path": "/muons_tight/2/eta",
            "hist_params": {
                "rebin": 4,
                "x_title": "#eta(#mu2)",
                "x_range": [-2.4, 2.4],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etae1": {
            "path": "/electrons_tight/1/eta",
            "hist_params": {
                "rebin": 5,
                "x_title": "#eta(e1)",
                "x_range": [-2.5, 2.5],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phimu1": {
            "path": "/muons_tight/1/phi",
            "hist_params": {
                "rebin": 4,
                "x_title": "#phi(#mu1)",
                "x_range": [-3.2, 3.2],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phimu2": {
            "path": "/muons_tight/2/phi",
            "hist_params": {
                "rebin": 4,
                "x_title": "#phi(#mu2)",
                "x_range": [-3.2, 3.2],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phie1": {
            "path": "/electrons_tight/1/phi",
            "hist_params": {
                "rebin": 4,
                "x_title": "#phi(e1)",
                "x_range": [-3.2, 3.2],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptj1": {
            "path": "/jets_cleaned/1/pt",
            "hist_params": {
                "rebin": 5,
                "x_title": "p_{T}(j1)",
                "x_range": [0., 200.],
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptj2": {
            "path": "/jets_cleaned/2/pt",
            "hist_params": {
                "rebin": 5,
                "x_title": "p_{T}(j2)",
                "x_range": [0., 200.],
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptj3": {
            "path": "/jets_cleaned/3/pt",
            "hist_params": {
                "rebin": 5,
                "x_title": "p_{T}(j3)",
                "x_range": [0., 200.],
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etaj1": {
            "path": "/jets_cleaned/1/eta",
            "hist_params": {
                "rebin": 6,
                "x_title": "#eta(j1)",
                "x_range": [-2.4, 2.4],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etaj2": {
            "path": "/jets_cleaned/2/eta",
            "hist_params": {
                "rebin": 6,
                "x_title": "#eta(j2)",
                "x_range": [-2.4, 2.4],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etaj3": {
            "path": "/jets_cleaned/3/eta",
            "hist_params": {
                "rebin": 6,
                "x_title": "#eta(j3)",
                "x_range": [-2.4, 2.4],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phij1": {
            "path": "/jets_cleaned/1/phi",
            "hist_params": {
                "rebin": 4,
                "x_title": "#phi(j1)",
                "x_range": [-3.2, 3.2],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phij2": {
            "path": "/jets_cleaned/2/phi",
            "hist_params": {
                "rebin": 4,
                "x_title": "#phi(j2)",
                "x_range": [-3.2, 3.2],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phij3": {
            "path": "/jets_cleaned/3/phi",
            "hist_params": {
                "rebin": 4,
                "x_title": "#phi(j3)",
                "x_range": [-3.2, 3.2],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "nJets": {
            "path": "/jets_cleaned/size",
            "hist_params": {
                "x_title": "N_{j}",
                "x_range": [0, 10],
                "y_title": "Events",
                "ratio_range": [0., 2.]
            }
        },
        "ptb1": {
            "path": "/bjets_cleaned/1/pt",
            "hist_params": {
                "rebin": 5,
                "x_title": "p_{T}(b1)",
                "x_range": [0., 200.],
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptb2": {
            "path": "/bjets_cleaned/2/pt",
            "hist_params": {
                "rebin": 5,
                "x_title": "p_{T}(b2)",
                "x_range": [0., 200.],
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "ptb3": {
            "path": "/bjets_cleaned/3/pt",
            "hist_params": {
                "rebin": 5,
                "x_title": "p_{T}(b3)",
                "x_range": [0., 200.],
                "y_title": "Events / 5 GeV",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etab1": {
            "path": "/bjets_cleaned/1/eta",
            "hist_params": {
                "rebin": 6,
                "x_title": "#eta(b1)",
                "x_range": [-2.4, 2.4],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etab2": {
            "path": "/bjets_cleaned/2/eta",
            "hist_params": {
                "rebin": 6,
                "x_title": "#eta(b2)",
                "x_range": [-2.4, 2.4],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "etab3": {
            "path": "/bjets_cleaned/3/eta",
            "hist_params": {
                "rebin": 6,
                "x_title": "#eta(b3)",
                "x_range": [-2.4, 2.4],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phib1": {
            "path": "/bjets_cleaned/1/phi",
            "hist_params": {
                "rebin": 4,
                "x_title": "#phi(b1)",
                "x_range": [-3.2, 3.2],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phib2": {
            "path": "/bjets_cleaned/2/phi",
            "hist_params": {
                "rebin": 4,
                "x_title": "#phi(b2)",
                "x_range": [-3.2, 3.2],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "phib3": {
            "path": "/bjets_cleaned/3/phi",
            "hist_params": {
                "rebin": 4,
                "x_title": "#phi(b3)",
                "x_range": [-3.2, 3.2],
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        },
        "nBJets": {
            "path": "/bjets_cleaned/size",
            "hist_params": {
                "x_title": "N_{b}",
                "x_range": [0, 10],
                "y_title": "Events",
                "ratio_range": [0., 2.]
            }
        },
        "METv": {
            "path": "/METv/pt",
            "hist_params": {
                "rebin": 5,
                "x_title": "E_{T}^{miss}",
                "x_range": [0., 200.],
                "y_title": "Events / 5 GeV",
                "ratio_range": [0., 2.0]
            }
        },
        "etaMETv": {
            "path": "/METv/eta",
            "hist_params": {
                "rebin": 4,
                "x_title": "#eta(E_{T}^{miss})",
                "y_title": "Events",
                "ratio_range": [0., 2.0]
            }
        },
        "phiMETv": {
            "path": "/METv/phi",
            "hist_params": {
                "rebin": 4,
                "x_title": "#phi(E_{T}^{miss})",
                "y_title": "Events",
                "ratio_range": [0.5, 1.5]
            }
        }
    }
}
