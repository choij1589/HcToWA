import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from itertools import product
from ctypes import c_double
from ROOT import TFile, TH2D
from Conversion.measConvSF import get_conv_sf

# outfile
outfile = TFile("outfile.root", "recreate")

# helper functions
def get_hist(sample, histkey, channel, mass_point, syst="Central"):
    if sample == "fake":
        fkey = f"Outputs/{channel}/{mass_point}/DATA.root"
    else:
        fkey = f"Outputs/{channel}/{mass_point}/{sample}.root"
    histkey = f"{sample}/SR/{syst}/{histkey}"

    f = TFile.Open(fkey)
    h = f.Get(histkey)
    h.SetDirectory(0)
    f.Close()

    return h

def optimize(MA, MHc, channel):
    print(f"MHc: {MHc}, MA: {MA}")
    histkey = 'mA_score'
    mass_point = f'MHc{MHc}_MA{MA}'
    signal = f"TTToHcToWA_AToMuMu_{mass_point}"
    out = list()

    # signal
    h_signal = get_hist(signal, histkey, channel, mass_point)
    
    # fake
    h_fake = get_hist('fake', histkey, channel, mass_point)
    h_fake_up = get_hist('fake', histkey, channel, mass_point, syst='Up')
    h_fake_down = get_hist('fake', histkey, channel, mass_point, syst='Down')

    # conversion
    DY_sf, DY_err = get_conv_sf(channel, 'DY')
    ZG_sf, ZG_err = get_conv_sf(channel, 'ZG')
    h_DY = get_hist('DY', histkey, channel, mass_point).Clone("DY")
    h_DY_up = get_hist('DY', histkey, channel, mass_point).Clone("DY_up")
    h_DY_down = get_hist('DY', histkey, channel, mass_point).Clone("DY_down")
    h_DY.Scale(DY_sf); h_DY_up.Scale(DY_sf + DY_err); h_DY_down.Scale(DY_sf - DY_err)
    h_ZG = get_hist('ZG', histkey, channel, mass_point).Clone("ZG")
    h_ZG_up = get_hist('ZG', histkey, channel, mass_point).Clone("ZG_up")
    h_ZG_down = get_hist('ZG', histkey, channel, mass_point).Clone("ZG_down")
    h_ZG.Scale(ZG_sf); h_ZG_up.Scale(ZG_sf + ZG_err); h_ZG_down.Scale(ZG_sf - ZG_err)

    h_conv = h_DY.Clone("conv"); h_conv.Add(h_ZG)
    h_conv_up = h_DY_up.Clone("conv_up"); h_conv_up.Add(h_conv_up)
    h_conv_down = h_DY_down.Clone("conv_down"); h_conv_down.Add(h_conv_down)

    # others
    h_ttX = get_hist('ttX', histkey, channel, mass_point)
    h_rare = get_hist('rare', histkey, channel, mass_point)
    h_VV = get_hist('VV', histkey, channel, mass_point)

    mass_steps = 151
    score_start = 0.
    score_steps = 100
    epsilon =  10e-6

    # estimate with mA only
    sign_opt = 0.
    m_step_opt = 0.
    s_step_opt = 0.
    N_sig_opt = 0.
    N_bkg_opt = 0.
    N_bkg_err_opt = 0.
    h = TH2D(f"MHc{MHc}_MA{MA}", "", 50, 0., 5., 100, 0., 1.)
    h_ROC_mA = TH2D(f"ROC_mA_MHc{MHc}_MA{MA}", "", 1000, 0., 1., 1000, 0., 1.)
    h_ROC_score = TH2D(f"ROC_score_MHc{MHc}_MA{MA}", "", 1000, 0., 1., 1000, 0., 1.)
    lst_eff_mA = list()
    lst_rej_mA = list()
    for i in range(1, mass_steps):
        m_step = 0.1*i
        mA_left = h_signal.GetXaxis().FindBin(MA - m_step + epsilon)
        mA_right = h_signal.GetXaxis().FindBin(MA + m_step + epsilon)
        mA_left_end = h_signal.GetXaxis().FindBin(0. - epsilon)
        mA_right_end = h_signal.GetXaxis().FindBin(300. + epsilon)
        score_left = h_signal.GetYaxis().FindBin(0. - epsilon)
        score_right = h_signal.GetYaxis().FindBin(1. + epsilon)

        # signal
        N_sig_err = c_double(0.)
        N_sig = h_signal.IntegralAndError(mA_left, mA_right, score_left, score_right, N_sig_err)
        N_sig_err = max(N_sig_err.value, 0)
        N_sig_base = h_signal.Integral(mA_left_end, mA_right_end, score_left, score_right)

        # fake
        N_fake = h_fake.Integral(mA_left, mA_right, score_left, score_right)
        N_fake_err = N_fake + np.power(h_fake_up.Integral(mA_left, mA_right, score_left, score_right), 2) + np.power(h_fake_down.Integral(mA_left, mA_right, score_left, score_right), 2)
        N_fake_err = max(N_fake_err, 0.)
        N_fake_err = np.sqrt(N_fake_err)
        N_fake_base = h_fake.Integral(mA_left_end, mA_right_end, score_left, score_right)

        # conv
        N_conv = h_conv.Integral(mA_left, mA_right, score_left, score_right)
        N_conv_err = N_conv + np.power(h_conv_up.Integral(mA_left, mA_right, score_left, score_right), 2) + np.power(h_conv_down.Integral(mA_left, mA_right, score_left, score_right), 2)
        N_conv_err = max(N_conv_err, 0.)
        N_conv_err = np.sqrt(N_conv_err)
        N_conv_base = h_conv.Integral(mA_left_end, mA_right_end, score_left, score_right)

        # others
        N_ttX_err = c_double(0.)
        N_ttX = h_ttX.IntegralAndError(mA_left, mA_right, score_left, score_right, N_ttX_err)
        N_ttX_err = max(N_ttX_err.value, 0.)
        N_ttX_base = h_ttX.Integral(mA_left_end, mA_right_end, score_left, score_right)
       
        N_VV_err = c_double(0.)
        N_VV = h_VV.IntegralAndError(mA_left, mA_right, score_left, score_right, N_VV_err)
        N_VV_err = max(N_VV_err.value, 0.)
        N_VV_base = h_VV.Integral(mA_left_end, mA_right_end, score_left, score_right)

        N_rare_err = c_double(0.)
        N_rare = h_rare.IntegralAndError(mA_left, mA_right, score_left, score_right, N_rare_err)
        N_rare_err = max(N_rare_err.value, 0.)
        N_rare_base = h_rare.Integral(mA_left_end, mA_right_end, score_left, score_right)

        N_bkg = N_fake + N_conv + N_ttX + N_VV + N_rare
        N_bkg_err = N_fake_err + N_conv_err + N_ttX_err + N_VV_err + N_rare_err
        N_bkg_base = N_fake_base + N_conv_base + N_ttX_base + N_VV_base + N_rare_base

        eff = N_sig / N_sig_base
        rej = 1. - (N_bkg / N_bkg_base)
        h_ROC_mA.Fill(eff, rej, 1.)
        lst_eff_mA.append(eff)
        lst_rej_mA.append(rej)

        sign = N_sig / np.sqrt(N_bkg + np.power(N_bkg_err, 2))
        # sign = N_sig / np.sqrt(N_bkg)
        if sign > sign_opt:
            sign_opt = sign
            m_step_opt = m_step
            N_sig_opt = N_sig
            N_bkg_opt = N_bkg
            N_bkg_opt_err = N_bkg_err
    
    out.append(sign_opt)
    out.append(N_sig_opt)
    out.append(N_bkg_opt)
    out.append(N_bkg_opt_err)
    out.append(m_step_opt)
    print(sign_opt, m_step_opt, N_sig_opt, N_bkg_opt, N_bkg_opt_err)

    lst_eff_score = list()
    lst_rej_score = list()
    for i in range(score_steps):
        s_step = 0.1*i
        mA_left = h_signal.GetXaxis().FindBin(0. - epsilon)
        mA_right = h_signal.GetXaxis().FindBin(300. + epsilon)
        score_left_end = h_signal.GetYaxis().FindBin(0. - epsilon)
        score_right_end = h_signal.GetYaxis().FindBin(1. + epsilon)
        score_left = h_signal.GetYaxis().FindBin(score_start + s_step - epsilon)
        score_right = h_signal.GetYaxis().FindBin(1. + epsilon)

        # signal
        N_sig_err = c_double(0.)
        N_sig = h_signal.IntegralAndError(mA_left, mA_right, score_left, score_right, N_sig_err)
        N_sig_err = max(N_sig_err.value, 0)
        N_sig_base = h_signal.Integral(mA_left, mA_right, score_left_end, score_right_end)

        # fake
        N_fake = h_fake.Integral(mA_left, mA_right, score_left, score_right)
        N_fake_err = N_fake + np.power(h_fake_up.Integral(mA_left, mA_right, score_left, score_right), 2) + np.power(h_fake_down.Integral(mA_left, mA_right, score_left, score_right), 2)
        N_fake_err = max(N_fake_err, 0.)
        N_fake_err = np.sqrt(N_fake_err)
        N_fake_base = h_fake.Integral(mA_left, mA_right, score_left_end, score_right_end)

        # conv
        N_conv = h_conv.Integral(mA_left, mA_right, score_left, score_right)
        N_conv_err = N_conv + np.power(h_conv_up.Integral(mA_left, mA_right, score_left, score_right), 2) + np.power(h_conv_down.Integral(mA_left, mA_right, score_left, score_right), 2)
        N_conv_err = max(N_conv_err, 0.)
        N_conv_err = np.sqrt(N_conv_err)
        N_conv_base = h_conv.Integral(mA_left, mA_right, score_left_end, score_right_end)

        # others
        N_ttX_err = c_double(0.)
        N_ttX = h_ttX.IntegralAndError(mA_left, mA_right, score_left, score_right, N_ttX_err)
        N_ttX_err = max(N_ttX_err.value, 0.)
        N_ttX_base = h_ttX.Integral(mA_left, mA_right, score_left_end, score_right_end)

        N_VV_err = c_double(0.)
        N_VV = h_VV.IntegralAndError(mA_left, mA_right, score_left, score_right, N_VV_err)
        N_VV_err = max(N_VV_err.value, 0.)
        N_VV_base = h_VV.Integral(mA_left, mA_right, score_left_end, score_right_end)

        N_rare_err = c_double(0.)
        N_rare = h_rare.IntegralAndError(mA_left, mA_right, score_left, score_right, N_rare_err)
        N_rare_err = max(N_rare_err.value, 0.)
        N_rare_base = h_rare.Integral(mA_left, mA_right, score_left_end, score_right_end)

        N_bkg = N_fake + N_conv + N_ttX + N_VV + N_rare
        N_bkg_err = N_fake_err + N_conv_err + N_ttX_err + N_VV_err + N_rare_err
        N_bkg_base = N_fake_base + N_conv_base + N_ttX_base + N_VV_base + N_rare_base

        eff = N_sig / N_sig_base
        rej = 1. - (N_bkg / N_bkg_base)
        h_ROC_score.Fill(eff, rej, 1.)
        lst_eff_score.append(eff)
        lst_rej_score.append(rej)

    # estimate using mA and score
    for i, j in product(range(1, mass_steps), range(score_steps)):
        m_step = i*0.1
        s_step = j*0.01
        mA_left = h_signal.GetXaxis().FindBin(MA - m_step + epsilon)
        mA_right = h_signal.GetXaxis().FindBin(MA + m_step + epsilon)
        score_left = h_signal.GetYaxis().FindBin(score_start + s_step + epsilon)
        score_right = h_signal.GetYaxis().FindBin(1.0 + epsilon)

        # signal
        N_sig_err = c_double(0.)
        N_sig = h_signal.IntegralAndError(mA_left, mA_right, score_left, score_right, N_sig_err)
        N_sig_err = max(N_sig_err.value, 0.)

        # fake
        N_fake = h_fake.Integral(mA_left, mA_right, score_left, score_right)
        N_fake_err = N_fake + np.power(h_fake_up.Integral(mA_left, mA_right, score_left, score_right), 2) + np.power(h_fake_down.Integral(mA_left, mA_right, score_left, score_right), 2)
        N_fake_err = max(N_fake_err, 0.)
        N_fake_err = np.sqrt(N_fake_err)

        # conv
        N_conv = h_conv.Integral(mA_left, mA_right, score_left, score_right)
        N_conv_err = N_conv + np.power(h_conv_up.Integral(mA_left, mA_right, score_left, score_right), 2) + np.power(h_conv_down.Integral(mA_left, mA_right, score_left, score_right), 2)
        N_conv_err = max(N_conv_err, 0.)
        N_conv_err = np.sqrt(N_conv_err)

        # others
        N_ttX_err = c_double(0.)
        N_ttX = h_ttX.IntegralAndError(mA_left, mA_right, score_left, score_right, N_ttX_err)
        N_ttX_err = max(N_ttX_err.value, 0.)

        N_VV_err = c_double(0.)
        N_VV = h_VV.IntegralAndError(mA_left, mA_right, score_left, score_right, N_VV_err)
        N_VV_err = max(N_VV_err.value, 0.)

        N_rare_err = c_double(0.)
        N_rare = h_rare.IntegralAndError(mA_left, mA_right, score_left, score_right, N_rare_err)
        N_rare_err = max(N_rare_err.value, 0.)

        N_bkg = N_fake + N_conv + N_ttX + N_VV + N_rare
        N_bkg_err = N_fake_err + N_conv_err + N_ttX_err + N_VV_err + N_rare_err

        sign = N_sig / np.sqrt(N_bkg + np.power(N_bkg_err, 2))
        #print(m_step, score_start+s_step, sign)
        h.Fill(m_step+epsilon, score_start+s_step+epsilon, sign)

        # sign = N_sig / np.sqrt(N_bkg)
        if sign > sign_opt and sign != np.inf and N_bkg > 0.:
            sign_opt = sign
            m_step_opt = m_step
            s_step_opt = score_start + s_step
            N_sig_opt = N_sig
            N_bkg_opt = N_bkg
            N_bkg_opt_err = N_bkg_err

    out.append(sign_opt)
    out.append(N_sig_opt)
    out.append(N_bkg_opt)
    out.append(N_bkg_opt_err)
    out.append(m_step_opt)
    out.append(s_step_opt)
    print(sign_opt, m_step_opt, s_step_opt, N_sig_opt, N_bkg_opt, N_bkg_opt_err)
    print()

    outfile.cd()
    h.Write()
    h_ROC_mA.Write()
    h_ROC_score.Write()

    plt.figure(figsize=(15, 15))
    plt.plot(lst_eff_mA, lst_rej_mA, 'r--', label="ROC using MA window")
    plt.plot(lst_eff_score, lst_rej_score, 'b--', label="ROC using score")
    plt.legend(loc='best')
    plt.title(f"MHc{MHc}_MA{MA}")
    plt.xlabel("signal efficiency")
    plt.ylabel("background rejection")
    plt.xticks([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.grid(True)

    plt.savefig(f"Outputs/{channel}/MHc{MHc}_MA{MA}/ROC.png")

    return out

sample = dict()
sample["MHc70_MA15"] = optimize(MHc=70, MA=15, channel="1E2Mu")
sample["MHc70_MA40"] = optimize(MHc=70, MA=40, channel="1E2Mu")
sample["MHc70_MA65"] = optimize(MHc=70, MA=65, channel="1E2Mu")
sample["MHc100_MA15"] = optimize(MHc=100, MA=15, channel="1E2Mu")
sample["MHc100_MA25"] = optimize(MHc=100, MA=25, channel="1E2Mu")
sample["MHc100_MA60"] = optimize(MHc=100, MA=60, channel="1E2Mu")
sample["MHc100_MA95"] = optimize(MHc=100, MA=95, channel="1E2Mu")
sample["MHc130_MA15"] = optimize(MHc=130, MA=15, channel="1E2Mu")
sample["MHc130_MA45"] = optimize(MHc=130, MA=45, channel="1E2Mu")
sample["MHc130_MA55"] = optimize(MHc=130, MA=55, channel="1E2Mu")
sample["MHc130_MA90"] = optimize(MHc=130, MA=90, channel="1E2Mu")
sample["MHc130_MA125"] = optimize(MHc=130, MA=125, channel="1E2Mu")
sample["MHc160_MA15"] = optimize(MHc=160, MA=15, channel="1E2Mu")
sample["MHc160_MA45"] = optimize(MHc=160, MA=45, channel="1E2Mu")
sample["MHc160_MA75"] = optimize(MHc=160, MA=75, channel="1E2Mu")
sample["MHc160_MA85"] = optimize(MHc=160, MA=85, channel="1E2Mu")
sample["MHc160_MA120"] = optimize(MHc=160, MA=120, channel="1E2Mu")
sample["MHc160_MA155"] = optimize(MHc=160, MA=155, channel="1E2Mu")
outfile.Close()

columns = ["sign1", "N_sig1", "N_bkg1", "N_bkg_err1", "window_size1", "sign2", "N_sig2", "N_bkg2", "N_bkg_err2", "window_size2", "score_cut2"]
df = pd.DataFrame(sample, index=columns)
df.to_csv("outfile.csv")
print(df)


