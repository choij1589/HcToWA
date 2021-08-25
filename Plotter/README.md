# Plotter
---
### Requirements
Plotter use python3-root, please confirm your system can use:
```
python3
import ROOT
exit()
```
---
### How to
Base setting is defined in PlotterBase.py
Extra setting is defined in $CHILDCLASS.py

You should make child class in $CHILDCLASS.py, which inherited from PlotterBase

Parameters are classified in three categories, cvs_params, hist_params, and info_params
which is defined in Parameters/$child_cass_params.py

In $CHILDCLASS.py, make a code to get histograms and set the output path
TODO: add legend settings and normalization settings

Finally, run
```
python3 $CHILDCLASS.py
```
