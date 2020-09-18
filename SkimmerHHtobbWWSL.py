import os
import sys

from bamboo.analysismodules import SkimmerModule
from bamboo import treefunctions as op
from bamboo.analysisutils import makePileupWeight

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)))) # Add scripts in this directory
from BaseHHtobbWW import BaseNanoHHtobbWW
from selectionDef import *
from highlevelLambdas import *

#===============================================================================================#
#                                 SkimmerHHtobbWW                                               #
#===============================================================================================#
class SkimmerNanoHHtobbWWSL(BaseNanoHHtobbWW,SkimmerModule):
    """ Plotter module: HH->bbW(->e/µ nu)W(->e/µ nu) histograms from NanoAOD """
    def __init__(self, args):
        super(SkimmerNanoHHtobbWWSL, self).__init__(args)

    def defineSkimSelection(self, t, noSel, sample=None, sampleCfg=None): 
        noSel = super(SkimmerNanoHHtobbWWSL,self).prepareObjects(t, noSel, sample, sampleCfg, "SL", forSkimmer=True)
        # For the Skimmer, SF must not use defineOnFirstUse -> segmentation fault

        era = sampleCfg['era'] 

        #self.datadrivenContributions = {} # Avoid all data-driven estimates

        # Initialize varsToKeep dict #
        varsToKeep = dict()  

        #---------------------------------------------------------------------------------------# 
        #                                     Selections                                        #
        #---------------------------------------------------------------------------------------#
        if not self.inclusive_sel:
            #----- Check arguments -----#
            lepton_level = ["Preselected","Fakeable","Tight","FakeExtrapolation"]
            jet_level    = ["Ak4","Ak8",
                            "LooseResolved0b3j","LooseResolved1b2j","LooseResolved2b1j",
                            "TightResolved0b4j","TightResolved1b3j","TightResolved2b2j",
                            "SemiBoostedHbbWtoJ","SemiBoostedHbbWtoJJ","SemiBoostedWjj","Boosted"]

            # Only one lepton_level must be in args and Only one jet_level must be in args
            if [boolean for (level,boolean) in self.args.__dict__.items() if level in lepton_level].count(True) != 1:
                raise RuntimeError("Only one of the lepton arguments must be used, check --help")
            if [boolean for (level,boolean) in self.args.__dict__.items() if level in jet_level].count(True) != 1:
                raise RuntimeError("Only one of the jet arguments must be used, check --help")

            if self.args.Channel not in ["El","Mu"]:
                raise RuntimeError("Channel must be either 'El' or 'Mu'")

            #----- Lepton selection -----#
            # Args are passed within the self #
            selLeptonDict = makeSingleLeptonSelection(self,noSel,use_dd=False)
            # makeSingleLeptonSelection returns dict -> value is list of two selections for 2 channels 
            # [0] -> we take the first and only key and value because restricted to one lepton selection
            selLeptonList = list(selLeptonDict.values())[0]
            if self.args.Channel == "El":
                selObj = selLeptonList[0] # First item of list is El selection
            if self.args.Channel == "Mu":
                selObj = selLeptonList[1] # Second item of list is Mu selection

            #----- Jet selection -----#
            # Since the selections in one line, we can use the non copy option of the selection to modify the selection object internally
            if any([self.args.__dict__[item] for item in ["Ak4","LooseResolved0b3j","LooseResolved1b2j","LooseResolved2b1j"]]):
                makeCoarseResolvedSelection(self,selObj,nJet=3) 
            if any([self.args.__dict__[item] for item in ["TightResolved0b4j","TightResolved1b3j","TightResolved2b2j"]]):
                makeCoarseResolvedSelection(self,selObj,nJet=4) 
            if any([self.args.__dict__[item] for item in ["Ak8","SemiBoostedHbbWtoJ","SemiBoostedHbbWtoJJ"]]):
                makeCoarseBoostedSelection(self,selObj) 
            if self.args.LooseResolved0b3j:
                makeExclusiveLooseResolvedJetComboSelection(self,selObj,nbJet=0)
            if self.args.LooseResolved1b2j:
                makeExclusiveLooseResolvedJetComboSelection(self,selObj,nbJet=1)
            if self.args.LooseResolved2b1j:
                makeExclusiveLooseResolvedJetComboSelection(self,selObj,nbJet=2)
            if self.args.TightResolved0b4j:
                makeExclusiveTightResolvedJetComboSelection(self,selObj,nbJet=0)
            if self.args.TightResolved1b3j:
                makeExclusiveTightResolvedJetComboSelection(self,selObj,nbJet=1)
            if self.args.TightResolved2b2j:
                makeExclusiveTightResolvedJetComboSelection(self,selObj,nbJet=2)
            if self.args.SemiBoostedHbbWtoJ:
                makeSemiBoostedHbbSelection(self,selObj,nNonb=1)
            if self.args.SemiBoostedHbbWtoJJ:
                makeSemiBoostedHbbSelection(self,selObj,nNonb=2)

        #---------------------------------------------------------------------------------------# 
        #                                 Synchronization tree                                  #
        #---------------------------------------------------------------------------------------#
        if self.args.Synchronization:
            # Event variables #
            varsToKeep["event"]             = None # Already in tree
            varsToKeep["run"]               = None # Already in tree 
            varsToKeep["ls"]                = t.luminosityBlock
            varsToKeep["n_presel_mu"]       = op.static_cast("UInt_t",op.rng_len(self.muonsPreSel))
            varsToKeep["n_fakeablesel_mu"]  = op.static_cast("UInt_t",op.rng_len(self.muonsFakeSel))
            varsToKeep["n_mvasel_mu"]       = op.static_cast("UInt_t",op.rng_len(self.muonsTightSel))
            varsToKeep["n_presel_ele"]      = op.static_cast("UInt_t",op.rng_len(self.electronsPreSel))
            varsToKeep["n_fakeablesel_ele"] = op.static_cast("UInt_t",op.rng_len(self.electronsFakeSel))
            varsToKeep["n_mvasel_ele"]      = op.static_cast("UInt_t",op.rng_len(self.electronsTightSel))
            varsToKeep["n_presel_ak4Jet"]   = op.static_cast("UInt_t",op.rng_len(self.ak4Jets))    
            varsToKeep["n_presel_ak8Jet"]   = op.static_cast("UInt_t",op.rng_len(self.ak8BJets))    
            varsToKeep["n_medium_ak4BJet"]  = op.static_cast("UInt_t",op.rng_len(self.ak4BJets))    
            varsToKeep["is_SR"]             = op.static_cast("UInt_t",op.OR(op.rng_len(self.ElElDileptonTightSel)>=1,
                                                                            op.rng_len(self.MuMuDileptonTightSel)>=1,
                                                                            op.rng_len(self.ElMuDileptonTightSel)>=1))
            varsToKeep["is_CR"]             = op.static_cast("UInt_t",op.OR(op.rng_len(self.ElElDileptonFakeExtrapolationSel)>=1,
                                                                            op.rng_len(self.MuMuDileptonFakeExtrapolationSel)>=1,
                                                                            op.rng_len(self.ElMuDileptonFakeExtrapolationSel)>=1))
            varsToKeep["is_ee"]             = op.static_cast("UInt_t",op.OR(op.rng_len(self.ElElDileptonTightSel)>=1, op.rng_len(self.ElElDileptonFakeExtrapolationSel)>=1))
            varsToKeep["is_mm"]             = op.static_cast("UInt_t",op.OR(op.rng_len(self.MuMuDileptonTightSel)>=1, op.rng_len(self.MuMuDileptonFakeExtrapolationSel)>=1))
            varsToKeep["is_em"]             = op.static_cast("UInt_t",op.OR(op.rng_len(self.ElMuDileptonTightSel)>=1, op.rng_len(self.ElMuDileptonFakeExtrapolationSel)>=1))
            varsToKeep["is_resolved"]       = op.switch(op.AND(op.rng_len(self.ak4Jets)>=2,op.rng_len(self.ak4BJets)>=1,op.rng_len(self.ak8BJets)==0), op.c_bool(True), op.c_bool(False))
            varsToKeep["is_boosted"]        = op.switch(op.rng_len(self.ak8BJets)>=1, op.c_bool(True), op.c_bool(False))


            # Triggers #
            varsToKeep['n_leadfakeableSel_ele']     = op.static_cast("UInt_t",op.rng_len(self.leadElectronsFakeSel))
            varsToKeep['n_leadfakeableSel_mu']      = op.static_cast("UInt_t",op.rng_len(self.leadMuonsFakeSel))
            varsToKeep["triggers"]                  = self.triggers
            varsToKeep["triggers_SingleElectron"]   = op.OR(*self.triggersPerPrimaryDataset['SingleElectron'])
            varsToKeep["triggers_SingleMuon"]       = op.OR(*self.triggersPerPrimaryDataset['SingleMuon'])
            varsToKeep["triggers_DoubleElectron"]   = op.OR(*self.triggersPerPrimaryDataset['DoubleEGamma'])
            varsToKeep["triggers_DoubleMuon"]       = op.OR(*self.triggersPerPrimaryDataset['DoubleMuon'])
            varsToKeep["triggers_MuonElectron"]     = op.OR(*self.triggersPerPrimaryDataset['MuonEG'])

            # Muons #
            for i in range(1,3): # 2 leading muons
                varsToKeep["mu{}_pt".format(i)]                    = op.switch(op.rng_len(self.muonsPreSel) >= i, self.muonsPreSel[i-1].pt, op.c_float(-9999., "float"))
                varsToKeep["mu{}_eta".format(i)]                   = op.switch(op.rng_len(self.muonsPreSel) >= i, self.muonsPreSel[i-1].eta, op.c_float(-9999.))
                varsToKeep["mu{}_phi".format(i)]                   = op.switch(op.rng_len(self.muonsPreSel) >= i, self.muonsPreSel[i-1].phi, op.c_float(-9999.))
                varsToKeep["mu{}_E".format(i)]                     = op.switch(op.rng_len(self.muonsPreSel) >= i, self.muonsPreSel[i-1].p4.E(), op.c_float(-9999., "float"))
                varsToKeep["mu{}_charge".format(i)]                = op.switch(op.rng_len(self.muonsPreSel) >= i, self.muonsPreSel[i-1].charge, op.c_int(-9999.))
                varsToKeep["mu{}_conept".format(i)]                = op.switch(op.rng_len(self.muonsPreSel) >= i, self.muon_conept[self.muonsPreSel[i-1].idx], op.c_float(-9999.))
                varsToKeep["mu{}_miniRelIso".format(i)]            = op.switch(op.rng_len(self.muonsPreSel) >= i, self.muonsPreSel[i-1].miniPFRelIso_all, op.c_float(-9999.))
                varsToKeep["mu{}_PFRelIso04".format(i)]            = op.switch(op.rng_len(self.muonsPreSel) >= i, self.muonsPreSel[i-1].pfRelIso04_all, op.c_float(-9999.))
                varsToKeep["mu{}_jetNDauChargedMVASel".format(i)]  = op.c_float(-9999.)
                varsToKeep["mu{}_jetPtRel".format(i)]              = op.switch(op.rng_len(self.muonsPreSel) >= i, self.muonsPreSel[i-1].jetPtRelv2, op.c_float(-9999.))
                varsToKeep["mu{}_jetRelIso".format(i)]             = op.switch(op.rng_len(self.muonsPreSel) >= i, self.muonsPreSel[i-1].jetRelIso, op.c_float(-9999.))
                varsToKeep["mu{}_jetDeepJet".format(i)]            = op.switch(op.rng_len(self.muonsPreSel) >= i, self.muonsPreSel[i-1].jet.btagDeepFlavB, op.c_float(-9999.))
                varsToKeep["mu{}_sip3D".format(i)]                 = op.switch(op.rng_len(self.muonsPreSel) >= i, self.muonsPreSel[i-1].sip3d, op.c_float(-9999.))
                varsToKeep["mu{}_dxy".format(i)]                   = op.switch(op.rng_len(self.muonsPreSel) >= i, self.muonsPreSel[i-1].dxy, op.c_float(-9999.))
                varsToKeep["mu{}_dxyAbs".format(i)]                = op.switch(op.rng_len(self.muonsPreSel) >= i, op.abs(self.muonsPreSel[i-1].dxy), op.c_float(-9999.))
                varsToKeep["mu{}_dz".format(i)]                    = op.switch(op.rng_len(self.muonsPreSel) >= i, self.muonsPreSel[i-1].dz, op.c_float(-9999.))
                varsToKeep["mu{}_segmentCompatibility".format(i)]  = op.switch(op.rng_len(self.muonsPreSel) >= i, self.muonsPreSel[i-1].segmentComp, op.c_float(-9999.))
                varsToKeep["mu{}_leptonMVA".format(i)]             = op.switch(op.rng_len(self.muonsPreSel) >= i, self.muonsPreSel[i-1].mvaTTH, op.c_float(-9999.))
                varsToKeep["mu{}_mediumID".format(i)]              = op.switch(op.rng_len(self.muonsPreSel) >= i, self.muonsPreSel[i-1].mediumId, op.c_float(-9999.,"Bool_t"))
                varsToKeep["mu{}_dpt_div_pt".format(i)]            = op.switch(op.rng_len(self.muonsPreSel) >= i, self.muonsPreSel[i-1].tunepRelPt, op.c_float(-9999.))  # Not sure
                varsToKeep["mu{}_isfakeablesel".format(i)]         = op.switch(op.rng_len(self.muonsPreSel) >= i, op.switch(self.lambda_muonFakeSel(self.muonsPreSel[i-1]), op.c_int(1), op.c_int(0)), op.c_int(-9999))
                varsToKeep["mu{}_ismvasel".format(i)]              = op.switch(op.rng_len(self.muonsPreSel) >= i, op.switch(op.AND(self.lambda_muonTightSel(self.muonsPreSel[i-1]), self.lambda_muonFakeSel(self.muonsPreSel[i-1])), op.c_int(1), op.c_int(0)), op.c_int(-9999)) # mvasel encompasses fakeablesel
                varsToKeep["mu{}_isGenMatched".format(i)]          = op.switch(op.rng_len(self.muonsPreSel) >= i, op.switch(self.lambda_is_matched(self.muonsPreSel[i-1]), op.c_int(1), op.c_int(0)), op.c_int(-9999))
                varsToKeep["mu{}_genPartFlav".format(i)]           = op.switch(op.rng_len(self.muonsPreSel) >= i, self.muonsPreSel[i-1].genPartFlav, op.c_int(-9999))
                varsToKeep["mu{}_FR".format(i)]                    = op.switch(op.rng_len(self.muonsPreSel) >= i, self.muonFR(self.muonsPreSel[i-1]), op.c_int(-9999))
                varsToKeep["mu{}_FRCorr".format(i)]                = op.switch(op.rng_len(self.muonsPreSel) >= i, self.lambda_FF_mu(self.muonsPreSel[i-1]), op.c_int(-9999))

            
            # Electrons #
            for i in range(1,3): # 2 leading electrons 
                varsToKeep["ele{}_pt".format(i)]                    = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].pt, op.c_float(-9999.))
                varsToKeep["ele{}_eta".format(i)]                   = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].eta, op.c_float(-9999.))
                varsToKeep["ele{}_phi".format(i)]                   = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].phi, op.c_float(-9999.))
                varsToKeep["ele{}_E".format(i)]                     = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].p4.E(), op.c_float(-9999., "float"))
                varsToKeep["ele{}_charge".format(i)]                = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].charge, op.c_int(-9999.))
                varsToKeep["ele{}_conept".format(i)]                = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electron_conept[self.electronsPreSel[i-1].idx], op.c_float(-9999.))
                varsToKeep["ele{}_miniRelIso".format(i)]            = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].miniPFRelIso_all, op.c_float(-9999.))
                varsToKeep["ele{}_PFRelIso03".format(i)]            = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].pfRelIso03_all, op.c_float(-9999.)) # Iso03, Iso04 not in NanoAOD
                varsToKeep["ele{}_jetNDauChargedMVASel".format(i)]  = op.c_float(-9999.)
                varsToKeep["ele{}_jetPtRel".format(i)]              = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].jetPtRelv2, op.c_float(-9999.))
                varsToKeep["ele{}_jetRelIso".format(i)]             = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].jetRelIso, op.c_float(-9999.))
                varsToKeep["ele{}_jetDeepJet".format(i)]            = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].jet.btagDeepFlavB, op.c_float(-9999.))
                varsToKeep["ele{}_sip3D".format(i)]                 = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].sip3d, op.c_float(-9999.))

                varsToKeep["ele{}_dxy".format(i)]                   = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].dxy, op.c_float(-9999.))
                varsToKeep["ele{}_dxyAbs".format(i)]                = op.switch(op.rng_len(self.electronsPreSel) >= i, op.abs(self.electronsPreSel[i-1].dxy), op.c_float(-9999.))
                varsToKeep["ele{}_dz".format(i)]                    = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].dz, op.c_float(-9999.))
                varsToKeep["ele{}_ntMVAeleID".format(i)]            = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].mvaFall17V2noIso, op.c_float(-9999.))
                varsToKeep["ele{}_leptonMVA".format(i)]             = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].mvaTTH, op.c_float(-9999.))
                varsToKeep["ele{}_passesConversionVeto".format(i)]  = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].convVeto, op.c_float(-9999.,"Bool_t"))
                varsToKeep["ele{}_nMissingHits".format(i)]          = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].lostHits, op.c_float(-9999.,"UChar_t"))
                varsToKeep["ele{}_sigmaEtaEta".format(i)]           = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].sieie, op.c_float(-9999.))
                varsToKeep["ele{}_HoE".format(i)]                   = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].hoe, op.c_float(-9999.))
                varsToKeep["ele{}_OoEminusOoP".format(i)]           = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].eInvMinusPInv, op.c_float(-9999.))
                varsToKeep["ele{}_isfakeablesel".format(i)]         = op.switch(op.rng_len(self.electronsPreSel) >= i, op.switch(self.lambda_electronFakeSel(self.electronsPreSel[i-1]), op.c_int(1), op.c_int(0)), op.c_int(-9999))
                varsToKeep["ele{}_ismvasel".format(i)]              = op.switch(op.rng_len(self.electronsPreSel) >= i, op.switch(op.AND(self.lambda_electronTightSel(self.electronsPreSel[i-1]), self.lambda_electronFakeSel(self.electronsPreSel[i-1])), op.c_int(1), op.c_int(0)), op.c_int(-9999)) # mvasel encompasses fakeablesel
                varsToKeep["ele{}_isGenMatched".format(i)]          = op.switch(op.rng_len(self.electronsPreSel) >= i, op.switch(self.lambda_is_matched(self.electronsPreSel[i-1]), op.c_int(1), op.c_int(0)), op.c_int(-9999))
                varsToKeep["ele{}_genPartFlav".format(i)]           = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].genPartFlav, op.c_int(-9999))
                varsToKeep["ele{}_deltaEtaSC".format(i)]            = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronsPreSel[i-1].deltaEtaSC, op.c_int(-9999))
                varsToKeep["ele{}_FR".format(i)]                    = op.switch(op.rng_len(self.electronsPreSel) >= i, self.electronFR(self.electronsPreSel[i-1]), op.c_int(-9999))
                varsToKeep["ele{}_FF".format(i)]                = op.switch(op.rng_len(self.electronsPreSel) >= i, self.lambda_FF_el(self.electronsPreSel[i-1]), op.c_int(-9999))

            # AK4 Jets #
            for i in range(1,5): # 4 leading jets 
                varsToKeep["ak4Jet{}_pt".format(i)]                 = op.switch(op.rng_len(self.ak4Jets) >= i, self.ak4Jets[i-1].pt, op.c_float(-9999.,"float"))
                varsToKeep["ak4Jet{}_eta".format(i)]                = op.switch(op.rng_len(self.ak4Jets) >= i, self.ak4Jets[i-1].eta, op.c_float(-9999.))
                varsToKeep["ak4Jet{}_phi".format(i)]                = op.switch(op.rng_len(self.ak4Jets) >= i, self.ak4Jets[i-1].phi, op.c_float(-9999.))
                varsToKeep["ak4Jet{}_E".format(i)]                  = op.switch(op.rng_len(self.ak4Jets) >= i, self.ak4Jets[i-1].p4.E(), op.c_float(-9999., "float"))
                varsToKeep["ak4Jet{}_CSV".format(i)]                = op.switch(op.rng_len(self.ak4Jets) >= i, self.ak4Jets[i-1].btagDeepFlavB, op.c_float(-9999.))
                varsToKeep["ak4Jet{}_hadronFlavour".format(i)]      = op.switch(op.rng_len(self.ak4Jets) >= i, self.ak4Jets[i-1].hadronFlavour, op.c_float(-9999.))
                varsToKeep["ak4Jet{}_btagSF".format(i)]             = op.switch(op.rng_len(self.ak4Jets) >= i, self.DeepJetDiscReshapingSF(self.ak4Jets[i-1]), op.c_float(-9999.))

            # AK8 Jets #
            for i in range(1,3): # 2 leading fatjets 
                varsToKeep["ak8Jet{}_pt".format(i)]                 = op.switch(op.rng_len(self.ak8BJets) >= i, self.ak8BJets[i-1].pt, op.c_float(-9999.))
                varsToKeep["ak8Jet{}_eta".format(i)]                = op.switch(op.rng_len(self.ak8BJets) >= i, self.ak8BJets[i-1].eta, op.c_float(-9999.))
                varsToKeep["ak8Jet{}_phi".format(i)]                = op.switch(op.rng_len(self.ak8BJets) >= i, self.ak8BJets[i-1].phi, op.c_float(-9999.))
                varsToKeep["ak8Jet{}_E".format(i)]                  = op.switch(op.rng_len(self.ak8BJets) >= i, self.ak8BJets[i-1].p4.E(), op.c_float(-9999., "float"))
                varsToKeep["ak8Jet{}_msoftdrop".format(i)]          = op.switch(op.rng_len(self.ak8BJets) >= i, self.ak8BJets[i-1].msoftdrop, op.c_float(-9999.))
                varsToKeep["ak8Jet{}_tau1".format(i)]               = op.switch(op.rng_len(self.ak8BJets) >= i, self.ak8BJets[i-1].tau1, op.c_float(-9999.))
                varsToKeep["ak8Jet{}_tau2".format(i)]               = op.switch(op.rng_len(self.ak8BJets) >= i, self.ak8BJets[i-1].tau2, op.c_float(-9999.))
                varsToKeep["ak8Jet{}_subjet0_pt".format(i)]         = op.switch(op.rng_len(self.ak8BJets) >= i, self.ak8BJets[i-1].subJet1.pt, op.c_float(-9999.))
                varsToKeep["ak8Jet{}_subjet0_eta".format(i)]        = op.switch(op.rng_len(self.ak8BJets) >= i, self.ak8BJets[i-1].subJet1.eta, op.c_float(-9999.))
                varsToKeep["ak8Jet{}_subjet0_phi".format(i)]        = op.switch(op.rng_len(self.ak8BJets) >= i, self.ak8BJets[i-1].subJet1.phi, op.c_float(-9999.))
                varsToKeep["ak8Jet{}_subjet0_CSV".format(i)]        = op.switch(op.rng_len(self.ak8BJets) >= i, self.ak8BJets[i-1].subJet1.btagDeepB, op.c_float(-9999.))
                varsToKeep["ak8Jet{}_subjet1_pt".format(i)]         = op.switch(op.rng_len(self.ak8BJets) >= i, self.ak8BJets[i-1].subJet2.pt, op.c_float(-9999.))
                varsToKeep["ak8Jet{}_subjet1_eta".format(i)]        = op.switch(op.rng_len(self.ak8BJets) >= i, self.ak8BJets[i-1].subJet2.eta, op.c_float(-9999.))
                varsToKeep["ak8Jet{}_subjet1_phi".format(i)]        = op.switch(op.rng_len(self.ak8BJets) >= i, self.ak8BJets[i-1].subJet2.phi, op.c_float(-9999.))
                varsToKeep["ak8Jet{}_subjet1_CSV".format(i)]        = op.switch(op.rng_len(self.ak8BJets) >= i, self.ak8BJets[i-1].subJet2.btagDeepB, op.c_float(-9999.))

            # MET #
             
            varsToKeep["PFMET"]    = self.corrMET.pt
            varsToKeep["PFMETphi"] = self.corrMET.phi

            # HME #

            # SF #
            from operator import mul
            from functools import reduce

            electronMuon_cont = op.combine((self.electronsFakeSel, self.muonsFakeSel))
            varsToKeep["trigger_SF"] = op.multiSwitch(
                    (op.AND(op.rng_len(self.electronsTightSel)==1,op.rng_len(self.muonsTightSel)==0) , self.ttH_singleElectron_trigSF(self.electronsTightSel[0])),
                    (op.AND(op.rng_len(self.electronsTightSel)==0,op.rng_len(self.muonsTightSel)==1) , self.ttH_singleMuon_trigSF(self.muonsTightSel[0])),
                    (op.AND(op.rng_len(self.electronsTightSel)>=2,op.rng_len(self.muonsTightSel)==0) , self.lambda_ttH_doubleElectron_trigSF(self.electronsTightSel)),
                    (op.AND(op.rng_len(self.electronsTightSel)==0,op.rng_len(self.muonsTightSel)>=2) , self.lambda_ttH_doubleMuon_trigSF(self.muonsTightSel)),
                    (op.AND(op.rng_len(self.electronsTightSel)>=1,op.rng_len(self.muonsTightSel)>=1) , self.lambda_ttH_electronMuon_trigSF(electronMuon_cont[0])),
                     op.c_float(1.))

            varsToKeep["lepton_IDSF"] = op.rng_product(self.electronsTightSel, lambda el : reduce(mul,self.lambda_ElectronLooseSF(el)+self.lambda_ElectronTightSF(el))) * \
                                        op.rng_product(self.muonsTightSel, lambda mu : reduce(mul,self.lambda_MuonLooseSF(mu)+self.lambda_MuonTightSF(mu))) 

            varsToKeep["lepton_IDSF_recoToLoose"] = op.rng_product(self.electronsTightSel, lambda el : reduce(mul,self.lambda_ElectronLooseSF(el))) * \
                                                    op.rng_product(self.muonsTightSel, lambda mu : reduce(mul,self.lambda_MuonLooseSF(mu)))
            varsToKeep["lepton_IDSF_looseToTight"] = op.rng_product(self.electronsTightSel, lambda el : reduce(mul,self.lambda_ElectronTightSF(el))) * \
                                                     op.rng_product(self.muonsTightSel, lambda mu : reduce(mul,self.lambda_MuonTightSF(mu)))

            # L1 Prefire #
            if era in ["2016","2017"]:
                varsToKeep["L1prefire"] = self.L1Prefiring
            else:
                varsToKeep["L1prefire"] = op.c_float(-9999.)

            # Fake rate #
            if self.args.FakeExtrapolation:
                varsToKeep["fakeRate"] = op.multiSwitch((op.rng_len(self.ElElDileptonFakeExtrapolationSel)>=1,self.ElElFakeFactor(self.ElElDileptonFakeExtrapolationSel[0])),
                                                        (op.rng_len(self.MuMuDileptonFakeExtrapolationSel)>=1,self.MuMuFakeFactor(self.MuMuDileptonFakeExtrapolationSel[0])),
                                                        (op.rng_len(self.ElMuDileptonFakeExtrapolationSel)>=1,self.ElMuFakeFactor(self.ElMuDileptonFakeExtrapolationSel[0])),
                                                        op.c_float(0.))
            else:
                varsToKeep["fakeRate"] = op.c_float(-9999.)

            # Btagging SF #
            varsToKeep["btag_SF"] = self.btagSF
            if "BtagRatioWeight" in self.__dict__.keys():
                varsToKeep["btag_reweighting"] = self.BtagRatioWeight
                varsToKeep["btag_reweighting_SF"] = self.btagSF * self.BtagRatioWeight

            # ttbar PT reweighting #
            if "group" in sampleCfg and sampleCfg["group"] == 'ttbar':
                varsToKeep["topPt_wgt"] = self.ttbar_weight(self.genTop[0],self.genAntitop[0])

           # Event Weight #
            if self.is_MC:
                #varsToKeep["MC_weight"] = op.sign(t.genWeight)
                varsToKeep["MC_weight"] = t.genWeight
                puWeightsFile = os.path.join(os.path.dirname(__file__), "data" , "pileup", sampleCfg["pufile"])
                varsToKeep["PU_weight"] = makePileupWeight(puWeightsFile, t.Pileup_nTrueInt, nameHint=f"puweightFromFile{sample}".replace('-','_'))
                varsToKeep["eventWeight"] = noSel.weight if self.inclusive_sel else selObj.sel.weight

           
            if self.inclusive_sel:
                return noSel, varsToKeep
            else:
                return selObj.sel, varsToKeep
                

        #---------------------------------------------------------------------------------------# 
        #                                    Selection tree                                     #
        #---------------------------------------------------------------------------------------#
        
        #----- EVT variables -----#
        varsToKeep["event"]     = None # Already in tree                                               
        varsToKeep["run"]       = None # Already in tree                                                                                                                                        
        varsToKeep["ls"]        = t.luminosityBlock

        #----- MET variables -----#
        MET = self.corrMET

        varsToKeep['METpt']   = MET.pt
        varsToKeep['METphi']  = MET.phi

        #----- Lepton variables -----#
        if self.args.Channel is None:
            raise RuntimeError("You need to specify --Channel")
        lepton = None
        if self.args.Preselected:
            if self.args.Channel == "El": lepton = self.electronsPreSel[0] 
            elif self.args.Channel == "Mu": lepton = self.muonsPreSel[0]
        if self.args.Fakeable:
            if self.args.Channel == "El": lepton = self.leadElectronFakeSel[0]
            elif self.args.Channel == "Mu": lepton = self.leadMuonFakeSel[0]
        if self.args.Tight:
            if self.args.Channel == "El": lepton = self.leadElectronTightSel[0]
            elif self.args.Channel == "Mu": lepton = self.leadMuonTightSel[0]
        if self.args.FakeExtrapolation:
            if self.args.Channel == "El": lepton = self.leadElectronFakeExtrapolationSel[0]
            elif self.args.Channel == "Mu": lepton = self.leadMuononFakeExtrapolationSel[0]

        varsToKeep['lep_Px']  = lepton.p4.Px()
        varsToKeep['lep_Py']  = lepton.p4.Py()
        varsToKeep['lep_Pz']  = lepton.p4.Pz()
        varsToKeep['lep_E']   = lepton.p4.E()
        varsToKeep['lep_pt']  = lepton.pt
        varsToKeep['lep_eta'] = lepton.eta
        varsToKeep['lep_phi'] = lepton.phi

        varsToKeep['lepmet_DPhi'] = op.abs(self.HLL.SinglepMet_dPhi(lepton,MET))
        varsToKeep['lepmet_pt']  = self.HLL.SinglepMet_Pt(lepton,MET)

        varsToKeep['lep_MT'] = self.HLL.MT(lepton,MET)
        varsToKeep['MET_LD'] = self.HLL.MET_LD(self.corrMET, self.ak4Jets, self.electronsFakeSel) if self.args.Channel == "El" else self.HLL.MET_LD(self.corrMET, self.ak4Jets, self.muonsFakeSel)
        varsToKeep['lep_conept'] = self.HLL.lambdaConePt(lepton)

        #----- Jet variables -----#
        if any([self.args.__dict__[item] for item in ["Ak4","LooseResolved0b3j","LooseResolved1b2j","LooseResolved2b1j",
                                                          "TightResolved0b4j","TightResolved1b3j","TightResolved2b2j"]]):
            if self.args.Ak4 or self.args.LooseResolved0b3j or self.args.TightResolved0b4j:
                jet1 = self.ak4Jets[0]
                jet2 = self.ak4Jets[1]
                jet3 = self.ak4Jets[2]
                if self.args.TightResolved0b4j:
                    jet4 = self.ak4Jets[3]

            if self.args.LooseResolved1b2j:
                jet1 = self.ak4BJets[0]
                jet2 = self.ak4LightJetsByBtagScore[0]
                jet3 = self.remainingJets[0] 

            if self.args.LooseResolved2b1j:
                jet1 = self.ak4BJets[0]
                jet2 = self.ak4BJets[1]
                jet3 = self.ak4LightJetsByPt[0]

            if self.args.TightResolved1b3j:
                jet1 = self.ak4BJets[0]
                jet2 = self.ak4LightJetsByBtagScore[0]
                
                lambda_chooseWjj_1b3j  = lambda dijet: op.abs((dijet[0].p4+dijet[1].p4+lepton.p4+self.corrMET.p4).M() - (self.HLL.bJetCorrP4(jet1) + self.HLL.bJetCorrP4(jet2)).M()) 
                WjjPairs_1b3j          = op.sort(self.remainingJetPairs(self.remainingJets), lambda_chooseWjj_1b3j)
                
                jet3 = WjjPairs_1b3j[0][0]
                jet4 = WjjPairs_1b3j[0][1]
            
            if self.args.TightResolved2b2j:
                jet1 = self.ak4BJets[0]
                jet2 = self.ak4BJets[1]
                
                lambda_chooseWjj_2b2j  = lambda dijet: op.abs((dijet[0].p4+dijet[1].p4+lepton.p4+self.corrMET.p4).M() - (self.HLL.bJetCorrP4(jet1) + self.HLL.bJetCorrP4(jet2)).M()) 
                WjjPairs_2b2j          = op.sort(self.remainingJetPairs(self.ak4LightJetsByPt), lambda_chooseWjj_2b2j)
                
                jet3 = WjjPairs_2b2j[0][0]
                jet4 = WjjPairs_2b2j[0][1]

            varsToKeep['nAk4Jets']  = op.static_cast("UInt_t",op.rng_len(self.ak4Jets))
            varsToKeep['nAk4BJets'] = op.static_cast("UInt_t",op.rng_len(self.ak4BJets))
            varsToKeep['j1_Px']  = self.HLL.bJetCorrP4(jet1).Px()
            varsToKeep['j1_Py']  = self.HLL.bJetCorrP4(jet1).Py()
            varsToKeep['j1_Pz']  = self.HLL.bJetCorrP4(jet1).Pz()
            varsToKeep['j1_E']   = self.HLL.bJetCorrP4(jet1).E()
            varsToKeep['j1_pt']  = self.HLL.bJetCorrP4(jet1).Pt()
            varsToKeep['j1_eta'] = self.HLL.bJetCorrP4(jet1).Eta()
            varsToKeep['j1_phi'] = self.HLL.bJetCorrP4(jet1).Phi()
            varsToKeep['j1_bTagDeepFlavB'] = jet1.btagDeepFlavB

            varsToKeep['j2_Px']  = self.HLL.bJetCorrP4(jet2).Px()
            varsToKeep['j2_Py']  = self.HLL.bJetCorrP4(jet2).Py()
            varsToKeep['j2_Pz']  = self.HLL.bJetCorrP4(jet2).Pz()
            varsToKeep['j2_E']   = self.HLL.bJetCorrP4(jet2).E()
            varsToKeep['j2_pt']  = self.HLL.bJetCorrP4(jet2).Pt()
            varsToKeep['j2_eta'] = self.HLL.bJetCorrP4(jet2).Eta()
            varsToKeep['j2_phi'] = self.HLL.bJetCorrP4(jet2).Phi()
            varsToKeep['j2_bTagDeepFlavB'] = jet2.btagDeepFlavB

            varsToKeep['j3_Px']  = jet3.p4.Px()
            varsToKeep['j3_Py']  = jet3.p4.Py()
            varsToKeep['j3_Pz']  = jet3.p4.Pz()
            varsToKeep['j3_E']   = jet3.p4.E()
            varsToKeep['j3_pt']  = jet3.pt
            varsToKeep['j3_eta'] = jet3.eta
            varsToKeep['j3_phi'] = jet3.phi
                                                                     
            # jet combo variables
            varsToKeep['j1j2_pt']   = (self.HLL.bJetCorrP4(jet1)+self.HLL.bJetCorrP4(jet2)).Pt()
            varsToKeep['j1j2_DR']   = op.deltaR(jet1.p4,jet2.p4)
            varsToKeep['j2j3_DR']   = op.deltaR(jet2.p4,jet3.p4)
            varsToKeep['j1j2_DPhi'] = op.abs(op.deltaPhi(jet1.p4,jet2.p4)) # Might need abs
            varsToKeep['j2j3_DPhi'] = op.abs(op.deltaPhi(jet2.p4,jet3.p4)) # Might need abs
            varsToKeep['j1j2_M']    = op.invariant_mass(self.HLL.bJetCorrP4(jet1),self.HLL.bJetCorrP4(jet2)) 
            varsToKeep['cosThetaS_Hbb'] = self.HLL.comp_cosThetaS(self.HLL.bJetCorrP4(jet1), self.HLL.bJetCorrP4(jet2))

            # highLevel variables
            varsToKeep['j1MetDPhi'] = op.abs(self.HLL.SinglepMet_dPhi(jet1, MET))
            varsToKeep['j2MetDPhi'] = op.abs(self.HLL.SinglepMet_dPhi(jet2, MET))
            varsToKeep['j3MetDPhi'] = op.abs(self.HLL.SinglepMet_dPhi(jet3, MET))

            varsToKeep['j1LepDR'] = op.deltaR(jet1.p4, lepton.p4)
            varsToKeep['j2LepDR'] = op.deltaR(jet2.p4, lepton.p4)
            varsToKeep['j3LepDR'] = op.deltaR(jet3.p4, lepton.p4)
            varsToKeep['j1LepDPhi'] = op.abs(op.deltaPhi(jet1.p4, lepton.p4))
            varsToKeep['j2LepDPhi'] = op.abs(op.deltaPhi(jet2.p4, lepton.p4))
            varsToKeep['j3LepDPhi'] = op.abs(op.deltaPhi(jet3.p4, lepton.p4))
            varsToKeep['minDR_lep_allJets'] = self.HLL.mindr_lep1_jet(lepton, self.ak4Jets)
            varsToKeep['mT_top_3particle']  = op.min(self.HLL.mT2(self.HLL.bJetCorrP4(jet1),lepton.p4 ,MET.p4), self.HLL.mT2(self.HLL.bJetCorrP4(jet2), lepton.p4, MET.p4))


            if self.args.LooseResolved0b3j or self.args.LooseResolved1b2j or self.args.LooseResolved2b1j:
                varsToKeep['minJetDR']       = self.HLL.MinDiJetDRLoose(jet1,jet2,jet3)
                varsToKeep['minLepJetDR']    = self.HLL.MinDR_lep3j(lepton,jet1,jet2,jet3)
                varsToKeep['HT2_lepJetMet']  = self.HLL.HT2_l3jmet(lepton,jet1,jet2,jet3,MET)
                varsToKeep['HT2R_lepJetMet'] = self.HLL.HT2R_l3jmet(lepton,jet1,jet2,jet3,MET)
                
            if self.args.TightResolved0b4j or self.args.TightResolved1b3j or self.args.TightResolved2b2j:
                varsToKeep['j4_Px']  = jet4.p4.Px()
                varsToKeep['j4_Py']  = jet4.p4.Py()
                varsToKeep['j4_Pz']  = jet4.p4.Pz()
                varsToKeep['j4_E']   = jet4.p4.E()
                varsToKeep['j4_pt']  = jet4.pt
                varsToKeep['j4_eta'] = jet4.eta
                varsToKeep['j4_phi'] = jet4.phi
                varsToKeep['j3j4_pt']   = (jet3.p4+jet4.p4).Pt() 
                varsToKeep['j3j4_DR']   = op.deltaR(jet3.p4,jet4.p4)
                varsToKeep['j3j4_DPhi'] = op.abs(op.deltaPhi(jet3.p4,jet4.p4))
                varsToKeep['j3j4_M']    = op.invariant_mass(jet3.p4, jet4.p4)
                varsToKeep['minJetDR']  = self.HLL.MinDiJetDRTight(jet1,jet2,jet3,jet4)
                varsToKeep['j4MetDPhi'] = op.abs(self.HLL.SinglepMet_dPhi(jet4, MET))
                varsToKeep['j4LepDR']   = op.deltaR(jet4.p4, lepton.p4)
                varsToKeep['j4LepDPhi'] = op.abs(op.deltaPhi(jet4.p4, lepton.p4))
                varsToKeep['minLepJetDR'] = self.HLL.MinDR_lep4j(lepton,jet1,jet2,jet3,jet4)
                varsToKeep['w1w2_MT']     = self.HLL.MT_W1W2_ljj(lepton,jet3,jet4,MET)
                varsToKeep['HT2_lepJetMet']         = self.HLL.HT2_l4jmet(lepton,jet1,jet2,jet3,jet4,MET)
                varsToKeep['HT2R_lepJetMet']        = self.HLL.HT2R_l4jmet(lepton,jet1,jet2,jet3,jet4,MET)

                varsToKeep['HWW_Mass'] = self.HLL.HWW_simple(jet3.p4,jet4.p4,lepton.p4,MET).M()
                varsToKeep['HWW_Simple_Mass'] = self.HLL.HWW_met_simple(jet3.p4,jet4.p4,lepton.p4,MET.p4).M()
                varsToKeep['HWW_dR'] = self.HLL.dR_Hww(jet3.p4,jet4.p4,lepton.p4,MET)
                varsToKeep['cosThetaS_Wjj_simple'] = self.HLL.comp_cosThetaS(jet3.p4, jet4.p4)
                varsToKeep['cosThetaS_WW_simple_met'] = self.HLL.comp_cosThetaS(self.HLL.Wjj_simple(jet3.p4,jet4.p4), self.HLL.Wlep_met_simple(lepton.p4, MET.p4))
                varsToKeep['cosThetaS_HH_simple_met'] = self.HLL.comp_cosThetaS(self.HLL.bJetCorrP4(jet1)+self.HLL.bJetCorrP4(jet2), self.HLL.HWW_met_simple(jet3.p4,jet4.p4,lepton.p4,MET.p4))

        #----- Fatjet variables -----#
        if any([self.args.__dict__[item] for item in ["Ak8","SemiBoostedHbbWtoJ","SemiBoostedHbbWtoJJ"]]):
            if self.args.Ak8:
                fatjet = self.ak8Jets[0]
            if self.args.Boosted:
                fatjet = self.ak8BJets[0]
            if self.args.SemiBoostedHbbWtoJ:
                fatjet = self.ak8BJets[0]
                jet2 = self.ak4JetsCleanedFromAk8b[0]
            if self.args.SemiBoostedHbbWtoJJ:
                fatjet = self.ak8BJets[0]

                lambda_chooseWjj_Hbb  = lambda dijet: op.abs((dijet[0].p4+dijet[1].p4+lepton.p4+self.corrMET.p4).M() - fatjet.p4.M()) 
                WjjPairs_Hbb          = op.sort(self.remainingJetPairs(self.ak4JetsCleanedFromAk8b), lambda_chooseWjj_Hbb)
                
                jet2 = WjjPairs_Hbb[0][0]
                jet3 = WjjPairs_Hbb[0][1]

            varsToKeep['fj_Px']  = fatjet.p4.Px()
            varsToKeep['fj_Py']  = fatjet.p4.Py()
            varsToKeep['fj_Pz']  = fatjet.p4.Pz()
            varsToKeep['fj_E']   = fatjet.p4.E()
            varsToKeep['fj_pt']  = fatjet.pt
            varsToKeep['fj_eta'] = fatjet.eta
            varsToKeep['fj_sub1pt']  = fatjet.subJet1.pt
            varsToKeep['fj_sub1eta'] = fatjet.subJet1.eta
            varsToKeep['fj_sub2pt']  = fatjet.subJet2.pt
            varsToKeep['fj_sub2eta'] = fatjet.subJet2.eta
            varsToKeep['fj_phi'] = fatjet.phi
            varsToKeep['fj_softdropMass'] = fatjet.msoftdrop
            # deepBtags
            varsToKeep['fj_btagDDBvL']        = fatjet.btagDDBvL
            varsToKeep['fj_btagDDBvL_noMD']   = fatjet.btagDDBvL_noMD
            varsToKeep['fj_btagDDCvB']        = fatjet.btagDDCvB
            varsToKeep['fj_btagDDCvB_noMD']   = fatjet.btagDDCvB_noMD
            varsToKeep['fj_btagDDCvL']        = fatjet.btagDDCvL
            varsToKeep['fj_btagDDCvL_noMD']   = fatjet.btagDDCvL_noMD
            varsToKeep['fj_btagDeepB']        = fatjet.btagDeepB
            
            varsToKeep['cosThetaS_Hbb'] = self.HLL.comp_cosThetaS(fatjet.subJet1.p4, fatjet.subJet2.p4)
            varsToKeep['mT_top_3particle']  = op.min(self.HLL.mT2(fatjet.subJet1.p4,lepton.p4,MET.p4), self.HLL.mT2(fatjet.subJet2.p4,lepton.p4,MET.p4))

            if self.args.SemiBoostedHbbWtoJ or self.args.SemiBoostedHbbWtoJJ:
                varsToKeep['fj_j2DR']   = op.deltaR(fatjet.p4, jet2.p4)
                varsToKeep['fj_j2DPhi'] = op.abs(op.deltaPhi(fatjet.p4, jet2.p4))
                varsToKeep['fjSub1_j2DR']   = op.deltaR(fatjet.subJet1.p4, jet2.p4)
                varsToKeep['fjSub1_j2DPhi'] = op.abs(op.deltaPhi(fatjet.subJet1.p4, jet2.p4))
                varsToKeep['fjSub2_j2DR']   = op.deltaR(fatjet.subJet2.p4, jet2.p4)
                varsToKeep['fjSub2_j2DPhi'] = op.abs(op.deltaPhi(fatjet.subJet2.p4, jet2.p4))
                varsToKeep['fj_lepDR']  = op.deltaR(fatjet.p4, lepton.p4)
                varsToKeep['fjSub1_lepDR']  = op.deltaR(fatjet.subJet1.p4, lepton.p4)
                varsToKeep['fjSub2_lepDR']  = op.deltaR(fatjet.subJet2.p4, lepton.p4)
                varsToKeep['fj_lepDPhi']  = op.abs(op.deltaPhi(fatjet.p4, lepton.p4))
                varsToKeep['fjSub1_lepDPhi']  = op.abs(op.deltaPhi(fatjet.subJet1.p4, lepton.p4))
                varsToKeep['fjSub2_lepDPhi']  = op.abs(op.deltaPhi(fatjet.subJet2.p4, lepton.p4))
                varsToKeep['minSubJetLepDR']  = op.min(op.deltaR(fatjet.subJet1.p4, lepton.p4), op.deltaR(fatjet.subJet2.p4, lepton.p4))

                if self.args.SemiBoostedHbbWtoJ:
                    varsToKeep['jetMinDR']   = self.HLL.MinDiJetDRLoose(fatjet.subJet1,fatjet.subJet2,jet2)
                    varsToKeep['jetLepMinDR'] = self.HLL.MinDR_lep3j(lepton,fatjet.subJet1,fatjet.subJet2,jet2) 
                    varsToKeep['HT2'] = self.HLL.HT2_l3jmet(lepton,fatjet.subJet1,fatjet.subJet2,jet2,MET)
                    varsToKeep['HT2R'] = self.HLL.HT2R_l3jmet(lepton,fatjet.subJet1,fatjet.subJet2,jet2,MET)
                    varsToKeep['MT_W1W2'] = self.HLL.MT_W1W2_lj(lepton,jet2,MET)

                if self.args.SemiBoostedHbbWtoJJ:
                    varsToKeep['Wtoj2j3_pt'] = (jet2.p4+jet3.p4).Pt()
                    varsToKeep['fj_j3DR']    = op.deltaR(fatjet.p4, jet3.p4)
                    varsToKeep['fj_j3DPhi']  = op.abs(op.deltaPhi(fatjet.p4, jet3.p4))
                    varsToKeep['fjSub1_j3DR']    = op.deltaR(fatjet.subJet1.p4, jet3.p4)
                    varsToKeep['fjSub1_j3DPhi']  = op.abs(op.deltaPhi(fatjet.subJet1.p4, jet3.p4))
                    varsToKeep['fjSub2_j3DR']    = op.deltaR(fatjet.subJet2.p4, jet3.p4)
                    varsToKeep['fjSub2_j3DPhi']  = op.abs(op.deltaPhi(fatjet.subJet2.p4, jet3.p4))
                    varsToKeep['j2_j3DR']    = op.deltaR(jet2.p4, jet3.p4)
                    varsToKeep['j2_j3DPhi']  = op.abs(op.deltaPhi(jet2.p4, jet3.p4))
                    varsToKeep['jetMinDR']   = self.HLL.MinDiJetDRTight(fatjet.subJet1,fatjet.subJet2,jet2,jet3)
                    varsToKeep['j2_j3invM']  = op.invariant_mass(jet2.p4,jet3.p4)
                    varsToKeep['j2_lepDR']   = op.deltaR(jet2.p4, lepton.p4)
                    varsToKeep['j3_lepDR']   = op.deltaR(jet3.p4, lepton.p4)
                    varsToKeep['j2_lepDPhi'] = op.abs(op.deltaPhi(jet2.p4, lepton.p4))
                    varsToKeep['j3_lepDPhi'] = op.abs(op.deltaPhi(jet3.p4, lepton.p4))
                    varsToKeep['jetLepMinDR'] = self.HLL.MinDR_lep4j(lepton,fatjet.subJet1,fatjet.subJet2,jet2,jet3) 
                    varsToKeep['HT2'] = self.HLL.HT2_l4jmet(lepton,fatjet.subJet1,fatjet.subJet2,jet2,jet3,MET)
                    varsToKeep['HT2R'] = self.HLL.HT2R_l4jmet(lepton,fatjet.subJet1,fatjet.subJet2,jet2,jet3,MET)
                    varsToKeep['MT_W1W2'] = self.HLL.MT_W1W2_ljj(lepton,jet2,jet3,MET)
                    varsToKeep['HWW_Mass'] = self.HLL.HWW_simple(jet2.p4,jet3.p4,lepton.p4,MET).M()
                    varsToKeep['HWW_Simple_Mass'] = self.HLL.HWW_met_simple(jet2.p4,jet3.p4,lepton.p4,MET.p4).M()
                    varsToKeep['HWW_dR'] = self.HLL.dR_Hww(jet2.p4,jet3.p4,lepton.p4,MET)
                    varsToKeep['cosThetaS_Wjj_simple'] = self.HLL.comp_cosThetaS(jet2.p4, jet3.p4)
                    varsToKeep['cosThetaS_WW_simple_met'] = self.HLL.comp_cosThetaS(self.HLL.Wjj_simple(jet2.p4,jet3.p4), self.HLL.Wlep_met_simple(lepton.p4, MET.p4))
                    varsToKeep['cosThetaS_HH_simple_met'] = self.HLL.comp_cosThetaS(fatjet.subJet1.p4+fatjet.subJet2.p4, self.HLL.HWW_met_simple(jet2.p4,jet3.p4,lepton.p4,MET.p4))
                    
        #----- Additional variables -----#
        varsToKeep["MC_weight"] = t.genWeight
        varsToKeep['total_weight'] = selObj.sel.weight

        #return leptonSel.sel, varsToKeep
        return selObj.sel, varsToKeep
