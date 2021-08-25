from ROOT import TLorentzVector


class Particle:
    def __init__(self, pt, eta, phi, mass):
        self.p4 = TLorentzVector()
        self.p4.SetPtEtaPhiM(pt, eta, phi, mass)

    def __add__(self, particle):
        temp = self.p4 + particle.P4()
        return Particle(temp.Pt(), temp.Eta(), temp.Phi(), temp.M())

    def __radd__(self, particle):
        temp = self.p4 + particle.P4()
        return Particle(temp.Pt(), temp.Eta(), temp.Phi(), temp.M())

    def P4(self):
        return self.p4

    def Pt(self):
        return self.p4.Pt()

    def Eta(self):
        return self.p4.Eta()

    def Phi(self):
        return self.p4.Phi()

    def Mass(self):
        return self.p4.M()

    def Energy(self):
        return self.p4.Energy()

    def DeltaR(self, particle):
        return self.p4.DeltaR(particle.P4())

    def DeltaPhi(self, particle):
        return self.p4.DeltaPhi(particle.P4())

# LeptonType
# 1: ewprompt
# 2: signal muons, from A
# 3: muons from tau
# 6: from offshell W, i.e. directly from Hc
# <0: fake leptons


class Lepton(Particle):
    def __init__(self, pt, eta, phi, mass):
        super().__init__(pt, eta, phi, mass)

    def SetCharge(self, charge):
        self.charge = charge

    def SetLepType(self, lepType):
        self.lepType = lepType

    def SetMiniIso(self, miniIso):
        self.miniIso = miniIso

    def SetRelIso(self, relIso):
        self.relIso = relIso

    def SetIsTight(self, isTight):
        self.isTight = isTight

    def Charge(self):
        return self.charge

    def LepType(self):
        return self.lepType

    def MiniIso(self):
        return self.miniIso

    def RelIso(self):
        return self.relIso

    def IsTight(self):
        return self.isTight


class Muon(Lepton):
    def __init__(self, pt, eta, phi, mass):
        super().__init__(pt, eta, phi, mass)

    def SetScaleUp(self, up):
        self.scaleUp = up

    def SetScaleDown(self, down):
        self.scaleDown = down

    def Scale(self, syst):
        if syst == "Up":
            self.p4 *= self.scaleUp
        elif syst == "Down":
            self.p4 *= self.scaleDown
        else:
            print("[DataFormat.Muon.Scale] syst should be Up or Down")
            raise(AttributeError)


class Electron(Lepton):
    def __init__(self, pt, eta, phi, mass):
        super().__init__(pt, eta, phi, mass)

    def SetScaleUp(self, up):
        self.scaleUp = up

    def SetScaleDown(self, down):
        self.scaleDown = down

    def SetSmearUp(self, up):
        self.smearUp = up

    def SetSmearDown(self, down):
        self.smearDown = down

    def Scale(self, syst):
        if syst == "Up":
            self.p4 *= self.scaleUp
        elif syst == "Down":
            self.p4 *= self.scaleDown
        else:
            print("[DataFormat.Electron.Scale] syst should be Up or Down")
            raise(AttributeError)

    def Smear(self, syst):
        if syst == "Up":
            self.p4 *= self.smearUp
        elif syst == "Down":
            self.p4 *= self.smearDown
        else:
            print("[DataFormat.Electron.Smear] syst should be Up or Down")
            raise(AttributeError)


class Jet(Particle):
    def __init__(self, pt, eta, phi, mass):
        super().__init__(pt, eta, phi, mass)
        
    def SetBtagScore(self, btagScore):
        self.btagScore = btagScore
        
    def BtagScore(self):
        return self.btagScore

    def SetIsBtagged(self, isBtagged):
        self.isBtagged = isBtagged

    def isBtagged(self):
        return self.isBtagged

    def SetScaleUp(self, up):
        self.scaleUp = up

    def SetScaleDown(self, down):
        self.scaleDown = down

    def SetSmearUp(self, up):
        self.smearUp = up

    def SetSmearDown(self, down):
        self.smearDown = down

    def Scale(self, syst):
        if syst == "Up":
            self.p4 *= self.scaleUp
        elif syst == "Down":
            self.p4 *= self.scaleDown
        else:
            print("[DataFormat.Jet.Scale] syst should be Up or Down")
            raise(AttributeError)

    def Smear(self, syst):
        if syst == "Up":
            self.p4 *= self.smearUp
        elif syst == "Down":
            self.p4 *= self.smearDown
        else:
            print("[DataFormat.Jet.Smear] syst should be Up or Down")
            raise(AttributeError)

# Useful functions


def get_leptons(evt):
    muons = []
    muons_zip = zip(evt.muons_pt,
                    evt.muons_eta,
                    evt.muons_phi,
                    evt.muons_mass,
                    evt.muons_charge,
                    evt.muons_lepType,
                    evt.muons_miniIso,
                    evt.muons_isTight,
                    evt.muons_scaleUp,
                    evt.muons_scaleDown
                    )
    for pt, eta, phi, mass, charge, lepType, miniIso, isTight, scaleUp, scaleDown in muons_zip:
        this_muon = Muon(pt, eta, phi, mass)
        this_muon.SetCharge(charge)
        this_muon.SetLepType(lepType)
        this_muon.SetMiniIso(miniIso)
        this_muon.SetIsTight(isTight)
        this_muon.SetScaleUp(scaleUp)
        this_muon.SetScaleDown(scaleDown)
        muons.append(this_muon)
    # check the number of muons
    if evt.nMuons != len(muons):
        Warning(f"muon entry is different, {evt.nMuons}: {len(muons)}")

    electrons = []
    electrons_zip = zip(evt.electrons_pt,
                        evt.electrons_eta,
                        evt.electrons_phi,
                        evt.electrons_mass,
                        evt.electrons_charge,
                        evt.electrons_lepType,
                        evt.electrons_miniIso,
                        evt.electrons_isTight,
                        evt.electrons_scaleUp,
                        evt.electrons_scaleDown,
                        evt.electrons_smearUp,
                        evt.electrons_smearDown
                        )
    for pt, eta, phi, mass, charge, lepType, miniIso, isTight, scaleUp, scaleDown, smearUp, smearDown in electrons_zip:
        this_electron = Electron(pt, eta, phi, mass)
        this_electron.SetCharge(charge)
        this_electron.SetLepType(lepType)
        this_electron.SetMiniIso(miniIso)
        this_electron.SetIsTight(isTight)
        this_electron.SetScaleUp(scaleUp)
        this_electron.SetScaleDown(scaleDown)
        this_electron.SetSmearUp(smearUp)
        this_electron.SetSmearDown(smearDown)
        electrons.append(this_electron)
    # check the number of electrons
    if evt.nElectrons != len(electrons):
        Warning(
            f"electron entry is different, {evt.nElectrons}: {len(electrons)}")

    return muons, electrons


def get_jets(evt):
    jets = []
    jets_zip = zip(evt.jets_pt,
                   evt.jets_eta,
                   evt.jets_phi,
                   evt.jets_mass,
                   evt.jets_btagScore,
                   evt.jets_isBtagged,
                   evt.jets_scaleUp,
                   evt.jets_scaleDown,
                   evt.jets_smearUp,
                   evt.jets_smearDown
                   )
    for pt, eta, phi, mass, btagScore, isBtagged, scaleUp, scaleDown, smearUp, smearDown in jets_zip:
        this_jet = Jet(pt, eta, phi, mass)
        this_jet.SetBtagScore(btagScore)
        this_jet.SetIsBtagged(isBtagged)
        this_jet.SetScaleUp(scaleUp)
        this_jet.SetScaleDown(scaleDown)
        this_jet.SetSmearUp(smearUp)
        this_jet.SetSmearDown(smearDown)
        jets.append(this_jet)
    # check the number of jets
    if evt.nJets != len(jets):
        Warning(f"jet entry is different, {evt.nJets}: {len(jets)}")

    bjets = []
    for jet in jets:
        if jet.isBtagged:
            bjets.append(jet)

    return jets, bjets


def scale_muons(muons, syst):
    for mu in muons:
        mu.Scale(syst)


def scale_electrons(electrons, syst):
    for ele in electrons:
        ele.Scale(syst)


def smear_electrons(electrons, syst):
    for ele in electrons:
        ele.Smear(syst)


def scale_jets(jets, syst):
    for jet in jets:
        jet.Scale(syst)


def smear_jets(jets, syst):
    for jet in jets:
        jet.Smear(syst)
