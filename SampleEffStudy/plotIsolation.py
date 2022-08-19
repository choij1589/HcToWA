from itertools import product
from ROOT import TFile
from ROOT import TH1D
from ROOT import TCanvas, TLegend

#### Arguments ####
ERAs = ["2016preVFP", "2016postVFP", "2017", "2018"]
ISO = "miniRelIso"      # relIso or miniRelIso
SIGNALs = ["MHc70_MA15", "MHc100_MA60", "MHc130_MA90", "MHc160_MA155"]
doTrkIso = False

#### Get files and histograms ####
def get_histograms(era, signal, doTrkIso):
    f_sig = TFile.Open(f"SKFlatOutput/{era}/Run3Mu__/SampleEffStudy_TTToHcToWA_AToMuMu_{signal}.root")
    f_bkg = TFile.Open(f"SKFlatOutput/{era}/Run3Mu__/SampleEffStudy_TTLL_powheg.root")

    if doTrkIso:
        h_prompt = f_sig.Get(f"3Mu/POGMedium+dZ+SIP+trkISO/PromptMuon/{ISO}")
        h_signal = f_sig.Get(f"3Mu/POGMedium+dZ+SIP+trkISO/SignalMuon/{ISO}")
        h_offshell = f_sig.Get(f"3Mu/POGMedium+dZ+SIP+trkISO/OffshellMuon/{ISO}")
        h_bkg = f_bkg.Get(f"3Mu/POGMedium+dZ+SIP+trkISO/FakeMuon/{ISO}")
    else:
        h_prompt = f_sig.Get(f"3Mu/POGMedium+dZ+SIP/PromptMuon/{ISO}")
        h_signal = f_sig.Get(f"3Mu/POGMedium+dZ+SIP/SignalMuon/{ISO}")
        h_offshell = f_sig.Get(f"3Mu/POGMedium+dZ+SIP/OffshellMuon/{ISO}")
        h_bkg = f_bkg.Get(f"3Mu/POGMedium+dZ+SIP/FakeMuon/{ISO}")
    h_prompt.SetDirectory(0)
    h_signal.SetDirectory(0)
    h_offshell.SetDirectory(0)
    h_bkg.SetDirectory(0)

    f_sig.Close()
    f_bkg.Close()

    return h_prompt, h_signal, h_offshell, h_bkg

#### Calculate efficiency ####
for era, sig in product(ERAs, SIGNALs):
    h_prompt, h_sig, h_offshell, h_bkg = get_histograms(era, sig, doTrkIso)
    h_sig.Add(h_prompt)
    h_sig.Add(h_offshell)

    h_eff_sig = TH1D("sig_eff", "", 100, 0., 1.)
    h_eff_bkg = TH1D("bkg_eff", "", 100, 0., 1.)

    for bin in range(h_eff_sig.GetNcells()):
        this_sig_eff = h_sig.Integral(0, bin) / h_sig.Integral(0, h_sig.GetNcells())
        this_bkg_eff = h_bkg.Integral(0, bin) / h_bkg.Integral(0, h_bkg.GetNcells())
        

        if bin == 10:
            print(f"{era} / {sig}: {this_sig_eff}, {this_bkg_eff}")
        h_eff_sig.SetBinContent(bin, this_sig_eff)
        h_eff_bkg.SetBinContent(bin, this_bkg_eff)

    h_eff_sig.SetStats(0)
    h_eff_sig.GetXaxis().SetLabelSize(0.05)
    h_eff_sig.GetXaxis().SetTitle(ISO)
    h_eff_sig.GetYaxis().SetTitle("Efficiency")
    h_eff_sig.GetYaxis().SetRangeUser(0., 1.)
    h_eff_sig.SetLineColor(1)
    h_eff_sig.SetLineWidth(2)

    h_eff_bkg.SetLineColor(2)
    h_eff_bkg.SetLineWidth(2)

    cvs = TCanvas()
    cvs.cd()
    h_eff_sig.Draw()
    h_eff_bkg.Draw("same")
    if doTrkIso:
        cvs.SaveAs(f"WithTrkIso/{era}_{sig}_{ISO}.png")
    else:
        cvs.SaveAs(f"NoTrkIso/{era}_{sig}_{ISO}.png")
    
colors = [1, 28, 9, 39]
for era in ERAs:
    h_eff_coll = {}
    for i, sig in enumerate(SIGNALs):
        h_prompt, h_sig, h_offshell, h_bkg = get_histograms(era, sig, doTrkIso)
        h_sig.Add(h_prompt)
        h_sig.Add(h_offshell)
        h_sig.SetDirectory(0)

        h_eff_sig = TH1D(f"sig_eff_{sig}", "", 100, 0., 1.)
        h_eff_bkg = TH1D(f"bkg_eff", "", 100, 0., 1.)

        for bin in range(h_eff_sig.GetNcells()):
            this_sig_eff = h_sig.Integral(0, bin) / h_sig.Integral(0, h_sig.GetNcells())
            this_bkg_eff = h_bkg.Integral(0, bin) / h_bkg.Integral(0, h_bkg.GetNcells())
            h_eff_sig.SetBinContent(bin, this_sig_eff)
            h_eff_bkg.SetBinContent(bin, this_bkg_eff)
        
        h_eff_sig.SetStats(0)
        h_eff_sig.GetXaxis().SetLabelSize(0.05)
        h_eff_sig.GetXaxis().SetTitle(ISO)
        h_eff_sig.GetYaxis().SetTitle("Efficiency")
        h_eff_sig.GetYaxis().SetRangeUser(0., 1.)
        h_eff_sig.SetLineColor(colors[i])
        h_eff_sig.SetLineWidth(2)

        h_eff_bkg.SetLineColor(2)
        h_eff_bkg.SetLineWidth(2)
    
        h_eff_sig.SetDirectory(0)
        h_eff_bkg.SetDirectory(0)
        h_eff_coll[sig] = h_eff_sig
        h_eff_coll["bkg"] = h_eff_bkg

    cvs = TCanvas()
    l = TLegend(0.5, 0.5, 0.9, 0.8)
    l.SetFillStyle(0)
    l.SetBorderSize(0)
    cvs.cd()
    for sig in SIGNALs:
        l.AddEntry(h_eff_coll[sig], sig, 'lep')
        h_eff_coll[sig].Draw("same")
    l.AddEntry(h_eff_coll['bkg'], "fake", "lep")
    h_eff_coll["bkg"].Draw("same")
    l.Draw("same")
    if doTrkIso: 
        cvs.SaveAs(f"WithTrkIso/{era}_{ISO}.png")
    else: 
        cvs.SaveAs(f"NoTrkIso/{era}_{ISO}.png")




