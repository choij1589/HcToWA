params = {
    # Z
    "ZMass": {
        "cvs_params": {
            "logy": False,
            "grid": True
        },
        "hist_params": {
            "rebin": 20,
            "x_title": "M(ll)",
            "y_title": "Events",
            "ratio_title": "x/Default",
            "ratio_range": [0.5, 1.5]
        },
        "info_params": {
            "info": "Normalized to #it{L}_{int} = 150 fb^{-1}",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
    "yZ": {
        "cvs_params": {
            "logy": False,
            "grid": True
        },
        "hist_params": {
            "x_title": "y^{Z}",
            "y_title": "Events",
            "ratio_title": "x/Default",
            "ratio_range": [0.5, 1.5],
        },
        "info_params": {
            "info": "Normalized to #it{L}_{int} = 150 fb^{-1}",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
    "ptZ": {
        "cvs_params": {
            "logy": True,
            "grid": True
        },
        "hist_params": {
            "rebin": 2,
            "x_title": "p_{T}(Z)",
            "y_title": "Events",
            "ratio_title": "x/Default",
            "x_range": [0., 200.],
            "ratio_range": [0.5, 1.5],
        },
        "info_params": {
            "info": "Normalized to #it{L}_{int} = 150 fb^{-1}",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
    "phiZ": {
        "cvs_params": {
            "logy": False,
            "grid": True
        },
        "hist_params": {
            "x_title": "#phi(Z)",
            "y_title": "Events",
            "ratio_title": "x/Default",
            "ratio_range": [0.5, 1.5],
        },
        "info_params": {
            "info": "Normalized to #it{L}_{int} = 150 fb^{-1}",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
    # Leptons
    "ptl1": {
        "cvs_params": {
            "logy": True,
            "grid": True
        },
        "hist_params": {
            "rebin": 2,
            "x_title": "p_{T}(l1)",
            "y_title": "Events",
            "ratio_title": "x/Default",
            "x_range": [0., 200.],
            "ratio_range": [0.5, 1.5],
        },
        "info_params": {
            "info": "Normalized to #it{L}_{int} = 150 fb^{-1}",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
    "ptl2": {
        "cvs_params": {
            "logy": True,
            "grid": True
        },
        "hist_params": {
            "rebin": 2,
            "x_title": "p_{T}(l2)",
            "y_title": "Events",
            "ratio_title": "x/Default",
            "x_range": [0., 200.],
            "ratio_range": [0.5, 1.5],
        },
        "info_params": {
            "info": "Normalized to #it{L}_{int} = 150 fb^{-1}",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
    "etal1": {
        "cvs_params": {
            "logy": False,
            "grid": True
        },
        "hist_params": {
            "x_title": "#eta(l1)",
            "y_title": "Events",
            "ratio_title": "x/Default",
            "ratio_range": [0.5, 1.5],
        },
        "info_params": {
            "info": "Normalized to #it{L}_{int} = 150 fb^{-1}",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
    "etal2": {
        "cvs_params": {
            "logy": False,
            "grid": True
        },
        "hist_params": {
            "x_title": "#eta(l2)",
            "y_title": "Events",
            "ratio_title": "x/Default",
            "ratio_range": [0.5, 1.5],
        },
        "info_params": {
            "info": "Normalized to #it{L}_{int} = 150 fb^{-1}",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
    "phil1": {
        "cvs_params": {
            "logy": False,
            "grid": True
        },
        "hist_params": {
            "x_title": "#phi(l1)",
            "y_title": "Events",
            "ratio_title": "x/Default",
            "ratio_range": [0.5, 1.5],
        },
        "info_params": {
            "info": "Normalized to #it{L}_{int} = 150 fb^{-1}",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
    "phil2": {
        "cvs_params": {
            "logy": False,
            "grid": True
        },
        "hist_params": {
            "x_title": "#phi(l2)",
            "y_title": "Events",
            "ratio_title": "x/Default",
            "ratio_range": [0.5, 1.5],
        },
        "info_params": {
            "info": "Normalized to #it{L}_{int} = 150 fb^{-1}",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
    "nLeptons": {
        "cvs_params": {
            "logy": False,
            "grid": True
        },
        "hist_params": {
            "x_title": "N(l)",
            "y_title": "Events",
            "ratio_title": "x/Default",
            "ratio_range": [0.5, 1.5],
        },
        "info_params": {
            "info": "Normalized to #it{L}_{int} = 150 fb^{-1}",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
    # Jets
    "ptj1": {
        "cvs_params": {
            "logy": True,
            "grid": True
        },
        "hist_params": {
            "x_title": "p_{T}(j1)",
            "y_title": "Events",
            "ratio_title": "x/Default",
            "x_range": [0., 200.],
            "ratio_range": [0., 2.0],
        },
        "info_params": {
            "info": "Normalized to #it{L}_{int} = 150 fb^{-1}",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
    "ptj2": {
        "cvs_params": {
            "logy": True,
            "grid": True
        },
        "hist_params": {
            "x_title": "p_{T}(j2)",
            "y_title": "Events",
            "ratio_title": "x/Default",
            "x_range": [0., 200.],
            "ratio_range": [0., 2.0],
        },
        "info_params": {
            "info": "Normalized to #it{L}_{int} = 150 fb^{-1}",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
    "etaj1": {
        "cvs_params": {
            "logy": False,
            "grid": True
        },
        "hist_params": {
            "x_title": "#eta(j1)",
            "y_title": "Events",
            "ratio_title": "x/Default",
            "ratio_range": [0., 2.0],
        },
        "info_params": {
            "info": "Normalized to #it{L}_{int} = 150 fb^{-1}",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
    "etaj2": {
        "cvs_params": {
            "logy": False,
            "grid": True
        },
        "hist_params": {
            "x_title": "#eta(j2)",
            "y_title": "Events",
            "ratio_title": "x/Default",
            "ratio_range": [0., 2.0],
        },
        "info_params": {
            "info": "Normalized to #it{L}_{int} = 150 fb^{-1}",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
    "phij1": {
        "cvs_params": {
            "logy": False,
            "grid": True
        },
        "hist_params": {
            "x_title": "#phi(j1)",
            "y_title": "Events",
            "ratio_title": "x/Default",
            "ratio_range": [0., 2.0],
        },
        "info_params": {
            "info": "Normalized to #it{L}_{int} = 150 fb^{-1}",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
    "phij2": {
        "cvs_params": {
            "logy": False,
            "grid": True
        },
        "hist_params": {
            "x_title": "#phi(j2)",
            "y_title": "Events",
            "ratio_title": "x/Default",
            "ratio_range": [0., 2.0],
        },
        "info_params": {
            "info": "Normalized to #it{L}_{int} = 150 fb^{-1}",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
    "nJets": {
        "cvs_params": {
            "logy": False,
            "grid": True
        },
        "hist_params": {
            "x_title": "N(j)",
            "y_title": "Events",
            "ratio_title": "x/Default",
            "ratio_range": [0.0, 2.0],
        },
        "info_params": {
            "info": "Normalized to #it{L}_{int} = 150 fb^{-1}",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
}
