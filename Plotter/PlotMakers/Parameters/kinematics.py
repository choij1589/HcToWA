params = {
    # Leading particle
    "1/pt": {
        "cvs_params": {
            "grid": True
        },
        "hist_params": {
            "rebin": 5,
            "x_title": "p_{T}(j1)",
            "x_range": [0., 300.],
            "y_title": "A.U.",
            "ratio_title": "x/Default",
            "ratio_range": [0., 2.0]
        },
        "info_params": {
            "info": "Normalized to 1",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
    # Subleading particle
    "2/pt": {
        "cvs_params": {
            "grid": True
        },
        "hist_params": {
            "rebin": 5,
            "x_title": "p_{T}(j2)",
            "x_range": [0., 200.],
            "y_title": "A.U.",
            "ratio_title": "x/Default",
            "ratio_range": [0., 2.0]
        },
        "info_params": {
            "info": "Normalized to 1",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
    # Third leading particle
    "3/pt": {
        "cvs_params": {
            "grid": True
        },
        "hist_params": {
            "rebin": 5,
            "x_title": "p_{T}(j3)",
            "x_range": [0., 160.],
            "y_title": "A.U.",
            "ratio_title": "x/Default",
            "ratio_range": [0., 2.0]
        },
        "info_params": {
            "info": "Normalized to 1",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
    # MVA
    "MVA": {
        "cvs_params": {
            "grid": True
        },
        "hist_params": {
            "x_title": "MVA score",
            "x_range": [-1., 1.],
            "y_title": "A.U.",
            "ratio_title": "x/Default",
            "ratio_range": [0., 2.0]
        },
        "info_params": {
            "info": "Normalized to 1",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    },
    # A
    "mass": {
        "cvs_params": {
             "grid": True
        },
        "hist_params": {
            "rebin": 2,
            "x_title": "M(#mu#mu)",
            "x_range": [0., 180.],
            "y_title": "A.U.",
            "ratio_title": "x/Default",
            "ratio_range": [0., 5.0]
        },
        "info_params": {
            "info": "Normalized to 1",
            "cms_text": "CMS",
            "extra_text": "Work on progress"
        }
    }
}
