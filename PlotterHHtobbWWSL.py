import os
import sys
from copy import copy

from itertools import chain

import logging
logger = logging.getLogger(__name__) 

import bamboo
from bamboo.analysismodules import HistogramsModule, DataDrivenBackgroundHistogramsModule

from bamboo import treefunctions as op
from bamboo.plots import CutFlowReport, Plot, EquidistantBinning, SummedPlot

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)))) # Add scripts in this directory
from BaseHHtobbWW import BaseNanoHHtobbWW
from plotDef import *
from selectionDef import *
from evaluateMVAs import *
from DDHelper import DataDrivenFake, DataDrivenDY


def switch_on_index(indexes, condition, contA, contB):
    if contA._base != contB._base:
        raise RuntimeError("The containers do not derive from the same base, this won't work")
    base = contA._base
    return [base[op.switch(condition, contA[index].idx, contB[index].idx)] for index in indexes]       

#===============================================================================================#
#                                       PlotterHHtobbWW                                         #
#===============================================================================================#
class PlotterNanoHHtobbWWSL(BaseNanoHHtobbWW,DataDrivenBackgroundHistogramsModule):
    """ Plotter module: HH->bbW(->e/µ nu)W(->e/µ nu) histograms from NanoAOD """
    def __init__(self, args):
        super(PlotterNanoHHtobbWWSL, self).__init__(args)

    def initialize(self):
        super(PlotterNanoHHtobbWWSL, self).initialize()
        # Change the way the FakeExtrapolation is postProcesses (avoids overriding the `postProcess` method) 
        if "FakeExtrapolation" in self.datadrivenContributions:
            contrib = self.datadrivenContributions["FakeExtrapolation"]
            self.datadrivenContributions["FakeExtrapolation"] = DataDrivenFake(contrib.name, contrib.config)
        if "DYEstimation" in self.datadrivenContributions: 
            contrib = self.datadrivenContributions["DYEstimation"]
            self.datadrivenContributions["DYEstimation"] = DataDrivenDY(contrib.name, contrib.config,"PseudoData" in self.datadrivenContributions)

    def definePlots(self, t, noSel, sample=None, sampleCfg=None): 
        noSel = super(PlotterNanoHHtobbWWSL,self).prepareObjects(t, noSel, sample, sampleCfg, 'SL')
        
        # --------------------------- Machine Learning Model --------------------------- #
        # Whad Tagger #
        path_bb1l_HH_XGB_Wjj_10Var_even = os.path.join(os.path.abspath(os.path.dirname(__file__)),'MachineLearning','bb1l_HH_XGB_Wjj_10Var_even.xml')
        path_bb1l_HH_XGB_Wjj_10Var_odd  = os.path.join(os.path.abspath(os.path.dirname(__file__)),'MachineLearning','bb1l_HH_XGB_Wjj_10Var_odd.xml')
        # SM - Event Level BDT
        path_fullRecoSM_even_simple_model   = os.path.join(os.path.abspath(os.path.dirname(__file__)),'MachineLearning','hh_bb1l_SM_Wjj_simple_full_reco_only_noIndPt_even.xml')
        path_missRecoSM_even_simple_model   = os.path.join(os.path.abspath(os.path.dirname(__file__)),'MachineLearning','hh_bb1l_SM_Wj1_even.xml')
        path_fullRecoSM_odd_simple_model    = os.path.join(os.path.abspath(os.path.dirname(__file__)),'MachineLearning','hh_bb1l_SM_Wjj_simple_full_reco_only_noIndPt_odd.xml')
        path_missRecoSM_odd_simple_model    = os.path.join(os.path.abspath(os.path.dirname(__file__)),'MachineLearning','hh_bb1l_SM_Wj1_odd.xml')
        path_fullRecoSM_even_BDT_model      = os.path.join(os.path.abspath(os.path.dirname(__file__)),'MachineLearning','hh_bb1l_SM_Wjj_BDT_full_reco_only_even.xml')
        path_fullRecoSM_odd_BDT_model       = os.path.join(os.path.abspath(os.path.dirname(__file__)),'MachineLearning','hh_bb1l_SM_Wjj_BDT_full_reco_only_odd.xml')
        # 900_Radion - Event Level BDT
        path_fullReco900R_even_simple_model = os.path.join(os.path.abspath(os.path.dirname(__file__)),'MachineLearning','hh_bb1l_X900GeV_Wjj_simple_full_reco_even.xml')
        path_missReco900R_even_simple_model = os.path.join(os.path.abspath(os.path.dirname(__file__)),'MachineLearning','hh_bb1l_X900GeV_Wj1_even.xml')
        path_fullReco900R_odd_simple_model  = os.path.join(os.path.abspath(os.path.dirname(__file__)),'MachineLearning','hh_bb1l_X900GeV_Wjj_simple_full_reco_odd.xml')
        path_missReco900R_odd_simple_model  = os.path.join(os.path.abspath(os.path.dirname(__file__)),'MachineLearning','hh_bb1l_X900GeV_Wj1_odd.xml')
        # DNN Model ResolvedFullReco #
        path_fullRecoResolved_SM_DNN_model_01 = os.path.join(os.path.abspath(os.path.dirname(__file__)),'MachineLearning','ml-models','DNN','FullRecoResolved','SM','TEST50_crossval0_1_model.pb')
        path_fullRecoResolved_SM_DNN_model_02 = os.path.join(os.path.abspath(os.path.dirname(__file__)),'MachineLearning','ml_models','DNN','FullRecoResolved','SM','TEST50_crossval1_2_model.pb')
        path_fullRecoResolved_SM_DNN_model_03 = os.path.join(os.path.abspath(os.path.dirname(__file__)),'MachineLearning','ml-models','DNN','FullRecoResolved','SM','TEST50_crossval2_3_model.pb')
        path_fullRecoResolved_SM_DNN_model_04 = os.path.join(os.path.abspath(os.path.dirname(__file__)),'MachineLearning','ml-models','DNN','FullRecoResolved','SM','TEST50_crossval3_4_model.pb')
        path_fullRecoResolved_SM_DNN_model_05 = os.path.join(os.path.abspath(os.path.dirname(__file__)),'MachineLearning','ml-models','DNN','FullRecoResolved','SM','TEST50_crossval4_5_model.pb')

        #if not os.path.exists(path_model):
            #raise RuntimeError('Could not find model file %s'%path_model)
        try:
            BDT_XGB_Wjj_even             = op.mvaEvaluator(path_bb1l_HH_XGB_Wjj_10Var_even, mvaType='TMVA')
            BDT_XGB_Wjj_odd              = op.mvaEvaluator(path_bb1l_HH_XGB_Wjj_10Var_odd, mvaType='TMVA')
            BDT_fullRecoSM_simple_even   = op.mvaEvaluator(path_fullRecoSM_even_simple_model, mvaType='TMVA')
            BDT_missRecoSM_simple_even   = op.mvaEvaluator(path_missRecoSM_even_simple_model, mvaType='TMVA')
            BDT_fullRecoSM_simple_odd    = op.mvaEvaluator(path_fullRecoSM_odd_simple_model, mvaType='TMVA')
            BDT_missRecoSM_simple_odd    = op.mvaEvaluator(path_missRecoSM_odd_simple_model, mvaType='TMVA')
            BDT_fullReco900R_simple_even = op.mvaEvaluator(path_fullReco900R_even_simple_model, mvaType='TMVA')
            BDT_missReco900R_simple_even = op.mvaEvaluator(path_missReco900R_even_simple_model, mvaType='TMVA')
            BDT_fullReco900R_simple_odd  = op.mvaEvaluator(path_fullReco900R_odd_simple_model, mvaType='TMVA')
            BDT_missReco900R_simple_odd  = op.mvaEvaluator(path_missReco900R_odd_simple_model, mvaType='TMVA')

            BDT_fullRecoSM_BDT_even   = op.mvaEvaluator(path_fullRecoSM_even_BDT_model, mvaType='TMVA')
            BDT_fullRecoSM_BDT_odd    = op.mvaEvaluator(path_fullRecoSM_odd_BDT_model, mvaType='TMVA')

            print (path_fullRecoResolved_SM_DNN_model_01)
            DNN_fullRecoResolved_SM_model_01 = op.mvaEvaluator(path_fullRecoResolved_SM_DNN_model_01, mvaType='Tensorflow',otherArgs=(['IN'], 'OUT/Softmax'))
            DNN_fullRecoResolved_SM_model_02 = op.mvaEvaluator(path_fullRecoResolved_SM_DNN_model_02, mvaType='Tensorflow',otherArgs=(['IN'], 'OUT/Softmax'))
            DNN_fullRecoResolved_SM_model_03 = op.mvaEvaluator(path_fullRecoResolved_SM_DNN_model_03, mvaType='Tensorflow',otherArgs=(['IN'], 'OUT/Softmax'))
            DNN_fullRecoResolved_SM_model_04 = op.mvaEvaluator(path_fullRecoResolved_SM_DNN_model_04, mvaType='Tensorflow',otherArgs=(['IN'], 'OUT/Softmax'))
            DNN_fullRecoResolved_SM_model_05 = op.mvaEvaluator(path_fullRecoResolved_SM_DNN_model_05, mvaType='Tensorflow',otherArgs=(['IN'], 'OUT/Softmax'))
        except:
            raise RuntimeError('Could not load model %s'%path_model)
        
        plots = []

        cutFlowPlots = []

        era = sampleCfg['era']

        self.sample = sample
        self.sampleCfg = sampleCfg
        self.era = era

        self.yieldPlots = makeYieldPlots(self.args.Synchronization)

        #----- Ratio reweighting variables (before lepton and jet selection) -----#
        if self.args.BtagReweightingOff or self.args.BtagReweightingOn:
            plots.append(objectsNumberPlot(channel="NoChannel",suffix='NoSelection',sel=noSel,objCont=self.ak4Jets,objName='Ak4Jets',Nmax=15,xTitle='N(Ak4 jets)'))
            plots.append(CutFlowReport("BtagReweightingCutFlowReport",noSel))
            return plots

        #----- Stitching study -----#
        if self.args.DYStitchingPlots or self.args.WJetsStitchingPlots:
            if self.args.DYStitchingPlots and sampleCfg['group'] != 'DY':
                raise RuntimeError("Stitching is only done on DY MC samples")
            if self.args.WJetsStitchingPlots and sampleCfg['group'] != 'Wjets':
                raise RuntimeError("Stitching is only done on WJets MC samples")
            plots.extend(makeLHEPlots(noSel,t.LHE))
            plots.append(objectsNumberPlot(channel="NoChannel",suffix='NoSelection',sel=noSel,objCont=self.ak4Jets,objName='Ak4Jets',Nmax=15,xTitle='N(Ak4 jets)'))
            plots.append(CutFlowReport("DYStitchingCutFlowReport",noSel))
            return plots


        #----- Singleleptons -----#
        selObjectDict = makeSingleLeptonSelection(self,noSel,plot_yield=True)
        # selObjectDict : keys -> level (str)
        #                 values -> [El,Mu] x Selection object
        # Select the jets selections that will be done depending on user input #
        jet_level = ["Ak4","Ak8","LooseResolved0b3j","LooseResolved1b2j","LooseResolved2b1j","TightResolved0b4j","TightResolved1b3j","TightResolved2b2j","SemiBoostedHbbWtoJ","SemiBoostedHbbWtoJJ","SemiBoostedWjj","Boosted"]
        jetplot_level = [arg for (arg,boolean) in self.args.__dict__.items() if arg in jet_level and boolean]
        if len(jetplot_level) == 0:  
            jetplot_level = jet_level # If nothing said, will do all
        jetsel_level = copy(jetplot_level)  # A plot level might need a previous selection that needs to be defined but not necessarily plotted
        if any("Resolved" in item for item in jetsel_level):     
            jetsel_level.append("Ak4") # Resolved needs the Ak4 selection
        if any("Boosted" in item for item in jetsel_level):     
            jetsel_level.append("Ak8") # SemiBoosted & Boosted needs the Ak8 selection

        # Selections:    
        # Loop over lepton selection and start plotting #
        for selectionType, selectionList in selObjectDict.items():
            print ("... Processing %s lepton type"%selectionType)
            #----- Select correct dilepton -----#
            if selectionType == "Preselected":  
                ElColl = self.electronsPreSel
                MuColl = self.muonsPreSel
            elif selectionType == "Fakeable":
                ElColl = self.leadElectronFakeSel
                MuColl = self.leadMuonFakeSel
            elif selectionType == "Tight":
                ElColl = [t.Electron[op.switch(op.rng_len(self.leadElectronTightSel) == 1, self.leadElectronTightSel[0].idx, 
                                                         self.leadElectronFakeExtrapolationSel[0].idx)]]
                MuColl = [t.Muon[op.switch(op.rng_len(self.leadMuonTightSel) == 1, self.leadMuonTightSel[0].idx, 
                                                     self.leadMuonFakeExtrapolationSel[0].idx)]]
            elif selectionType == "FakeExtrapolation":
                ElColl = self.leadElectronFakeExtrapolationSel
                MuColl = self.leadMuonFakeExtrapolationSel

            #----- Separate selections ------#
            ElSelObj = selectionList[0]
            MuSelObj = selectionList[1]

            if not self.args.OnlyYield:
                ChannelDictList = []
                ChannelDictList.append({'channel':'El','sel':ElSelObj.sel,'suffix':ElSelObj.selName})
                ChannelDictList.append({'channel':'Mu','sel':MuSelObj.sel,'suffix':MuSelObj.selName})
                
                for channelDict in ChannelDictList:
                    #----- Trigger plots -----#
                    plots.extend(singleLeptonTriggerPlots(**channelDict, triggerDict=self.triggersPerPrimaryDataset))
                    #----- Lepton plots -----#
                    # Singlelepton channel plots #
                    #plots.extend(singleLeptonChannelPlot(**channelDict, SinlepEl=ElColl, SinlepMu=MuColl, suffix=ElSelObj.selName))
        
            '''
            # ------------- test ------------- #
            testDictList=[]
            testKeys = ['channel','sel','jet','suffix']
            _ElSelObj = copy(ElSelObj)
            _MuSelObj = copy(MuSelObj)
            _ElSelObj.selName    += 'hasMinOneAk8Jet'
            _ElSelObj.yieldTitle += " + at least one Ak8"
            _ElSelObj.refine(cut = [op.rng_len(self.ak8Jets) >= 1], weight = None)
            _MuSelObj.selName    += 'hasMinOneAk8Jet'
            _MuSelObj.yieldTitle += " + at least one Ak8"
            _MuSelObj.refine(cut = [op.rng_len(self.ak8Jets) >= 1], weight = None)
            testDictList.append({'channel':'El','sel':_ElSelObj.sel,'jet':self.ak8Jets[0],'suffix':_ElSelObj.selName})
            testDictList.append({'channel':'Mu','sel':_MuSelObj.sel,'jet':self.ak8Jets[0],'suffix':_MuSelObj.selName})
            for testDict in testDictList:
                plots.extend(makeBtagDDplots(**{k:testDict[k] for k in testKeys}))
            #--------------------------------- #
            '''
            #----- Ak4 jets selection -----#
            LeptonKeys  = ['channel','sel','lep','suffix','is_MC']
            JetKeys     = ['channel','sel','j1','j2','j3','j4','suffix','nJet','nbJet','is_MC']
            commonItems = ['channel','sel','suffix']
            
            if "Ak4" in jetsel_level:
                print("... Processing Ak4Jets Selection for Resolved category")
                ChannelDictList = []
                JetsN    = {'objName':'Ak4Jets','objCont':self.ak4Jets,'Nmax':10,'xTitle':'N(Ak4 jets)'}
                FatJetsN = {'objName':'Ak8Jets','objCont':self.ak8Jets,'Nmax':5,'xTitle':'N(Ak8 jets)'}

                if any("LooseResolved" in key for key in jetsel_level):
                    print ("...... Processing Ak4 jet selection Loose (nAk4Jets = 3)")
                    ElSelObjAk4JetsLoose = makeCoarseResolvedSelection(self,ElSelObj,nJet=3,copy_sel=True,plot_yield=True)
                    MuSelObjAk4JetsLoose = makeCoarseResolvedSelection(self,MuSelObj,nJet=3,copy_sel=True,plot_yield=True)
                    if not self.args.OnlyYield:
                        # cutFlow Report #
                        cutFlowPlots.append(CutFlowReport(ElSelObjAk4JetsLoose.selName, ElSelObjAk4JetsLoose.sel))
                        cutFlowPlots.append(CutFlowReport(MuSelObjAk4JetsLoose.selName, MuSelObjAk4JetsLoose.sel))

                        if "Ak4" in jetplot_level and any("LooseResolved" in key for key in jetplot_level):
                            ChannelDictList.append({'channel':'El','sel':ElSelObjAk4JetsLoose.sel,'sinlepton':ElColl[0],
                                                    'j1':self.ak4Jets[0],'j2':self.ak4Jets[1],'j3':self.ak4Jets[2],'j4':None,
                                                    'nJet':3,'nbJet':0,
                                                    'suffix':ElSelObjAk4JetsLoose.selName,
                                                    'is_MC':self.is_MC})
                            ChannelDictList.append({'channel':'Mu','sel':MuSelObjAk4JetsLoose.sel,'sinlepton':MuColl[0],
                                                    'j1':self.ak4Jets[0],'j2':self.ak4Jets[1],'j3':self.ak4Jets[2],'j4':None,
                                                    'nJet':3,'nbJet':0,
                                                    'suffix':MuSelObjAk4JetsLoose.selName,
                                                    'is_MC':self.is_MC})

                if any("TightResolved" in key for key in jetsel_level):        
                    print ("...... Processing Ak4 jet selection Tight (nAk4Jets >= 4)")
                    ElSelObjAk4JetsTight = makeCoarseResolvedSelection(self,ElSelObj,nJet=4,copy_sel=True,plot_yield=True)
                    MuSelObjAk4JetsTight = makeCoarseResolvedSelection(self,MuSelObj,nJet=4,copy_sel=True,plot_yield=True)

                    # Jet and lepton plots #
                    if not self.args.OnlyYield:
                        # cutFlow Report #
                        cutFlowPlots.append(CutFlowReport(ElSelObjAk4JetsTight.selName, ElSelObjAk4JetsTight.sel))
                        cutFlowPlots.append(CutFlowReport(MuSelObjAk4JetsTight.selName, MuSelObjAk4JetsTight.sel))
                        if "Ak4" in jetplot_level and any("TightResolved" in key for key in jetplot_level):
                            ChannelDictList.append({'channel':'El','sel':ElSelObjAk4JetsTight.sel,'sinlepton':ElColl[0],
                                                    'j1':self.ak4Jets[0],'j2':self.ak4Jets[1],'j3':self.ak4Jets[2],'j4':self.ak4Jets[3],
                                                    'nJet':4,'nbJet':0,
                                                    'suffix':ElSelObjAk4JetsTight.selName,
                                                    'is_MC':self.is_MC})
                            ChannelDictList.append({'channel':'Mu','sel':MuSelObjAk4JetsTight.sel,'sinlepton':MuColl[0],
                                                    'j1':self.ak4Jets[0],'j2':self.ak4Jets[1],'j3':self.ak4Jets[2],'j4':self.ak4Jets[3],
                                                    'nJet':4,'nbJet':0,
                                                    'suffix':MuSelObjAk4JetsTight.selName,
                                                    'is_MC':self.is_MC})

                for channelDict in ChannelDictList:
                    # Dilepton #
                    plots.extend(makeSinleptonPlots(**{k:channelDict[k] for k in LeptonKeys}))
                    # Number of jets #
                    plots.append(objectsNumberPlot(**{k:channelDict[k] for k in commonItems},**JetsN))
                    plots.append(objectsNumberPlot(**{k:channelDict[k] for k in commonItems},**FatJetsN))
                    # Ak4 Jets #
                    plots.extend(makeAk4JetsPlots(**{k:channelDict[k] for k in JetKeys},HLL=self.HLL))
                    # MET #
                    plots.extend(makeMETPlots(**{k:channelDict[k] for k in commonItems}, met=self.corrMET))

            ##### Ak8-b jets selection #####
            if "Ak8" in jetsel_level:
                print ("...... Processing Ak8b jet selection for SemiBoosted & Boosted Category")
                ElSelObjAk8bJets = makeCoarseBoostedSelection(self,ElSelObj,copy_sel=True,plot_yield=True)
                MuSelObjAk8bJets = makeCoarseBoostedSelection(self,MuSelObj,copy_sel=True,plot_yield=True)

                FatJetKeys = ['channel','sel','j1','j2','j3','has1fat','suffix']
                FatJetsN   = {'objName':'Ak8Jets','objCont':self.ak8Jets,'Nmax':5,'xTitle':'N(Ak8 jets)'}
                SlimJetsN  = {'objName':'Ak4Jets','objCont':self.ak4Jets,'Nmax':10,'xTitle':'N(Ak4 jets)'}

                # Fatjets plots #
                ChannelDictList = []
                if not self.args.OnlyYield:
                    # cutFlow Report #
                    cutFlowPlots.append(CutFlowReport(ElSelObjAk8bJets.selName, ElSelObjAk8bJets.sel))
                    cutFlowPlots.append(CutFlowReport(MuSelObjAk8bJets.selName, MuSelObjAk8bJets.sel))
                    if "Ak8" in jetplot_level:
                        ChannelDictList.append({'channel':'El','sel':ElSelObjAk8bJets.sel,'sinlepton':ElColl[0],
                                                'j1':self.ak8BJets[0],'j2':None,'j3':None,'has1fat':True,
                                                'suffix':ElSelObjAk8bJets.selName,'is_MC':self.is_MC})
                        ChannelDictList.append({'channel':'Mu','sel':MuSelObjAk8bJets.sel,'sinlepton':MuColl[0],
                                                'j1':self.ak8BJets[0],'j2':None,'j3':None,'has1fat':True,
                                                'suffix':MuSelObjAk8bJets.selName,'is_MC':self.is_MC})

                for channelDict in ChannelDictList:
                    # Dilepton #
                    plots.extend(makeSinleptonPlots(**{k:channelDict[k] for k in LeptonKeys}))
                    # Number of jets #
                    plots.append(objectsNumberPlot(**{k:channelDict[k] for k in commonItems},**FatJetsN))
                    plots.append(objectsNumberPlot(**{k:channelDict[k] for k in commonItems},**SlimJetsN))
                    # Ak8 Jets #
                    plots.extend(makeSingleLeptonAk8JetsPlots(**{k:channelDict[k] for k in FatJetKeys},nMedBJets=self.nMediumBTaggedSubJets))
                    # MET #
                    plots.extend(makeMETPlots(**{k:channelDict[k] for k in commonItems}, met=self.corrMET))

                         

            # Used by TightResolved and Semi Boosted categories
            if self.args.WhadTagger == 'BDT':
                self.lambda_evaluateWhadBDT_el = lambda wjjPair : applyBDTforWhadTagger(ElColl[0], self.ak4Jets, self.ak4BJets, wjjPair,
                                                                                        BDT_XGB_Wjj_even, BDT_XGB_Wjj_odd, t.event)
                self.lambda_evaluateWhadBDT_mu = lambda wjjPair : applyBDTforWhadTagger(MuColl[0], self.ak4Jets, self.ak4BJets, wjjPair,
                                                                                        BDT_XGB_Wjj_even, BDT_XGB_Wjj_odd, t.event)


            #-----------------------------|||||||||||||||| Resolved selection ||||||||||||||-----------------------------------#
            if any("Resolved" in item for item in jetsel_level):
                ResolvedKeys = ['channel','sel','met','lep','j1','j2','j3','j4','suffix','nJet','nbJet']
                ChannelDictList = []
                ChannelDictListML = []

                # Resolved Selection (Loose) #
                #----- Resolved selection : 0 Btag -----#
                if "LooseResolved0b3j" in jetsel_level:
                    print ("......... Processing Loose Resolved jet (0 btag i.e. bTaggedJets = 0 & nLightJets = 3) selection")
                    ElSelObjAk4JetsLooseExclusiveResolved0b3j = makeExclusiveLooseResolvedJetComboSelection(self,ElSelObjAk4JetsLoose,nbJet=0,copy_sel=True,plot_yield=True)
                    MuSelObjAk4JetsLooseExclusiveResolved0b3j = makeExclusiveLooseResolvedJetComboSelection(self,MuSelObjAk4JetsLoose,nbJet=0,copy_sel=True,plot_yield=True)

                    if not self.args.OnlyYield and "LooseResolved0b3j" in jetplot_level:
                        # Cut flow report #
                        cutFlowPlots.append(CutFlowReport(ElSelObjAk4JetsLooseExclusiveResolved0b3j.selName,ElSelObjAk4JetsLooseExclusiveResolved0b3j.sel))
                        cutFlowPlots.append(CutFlowReport(MuSelObjAk4JetsLooseExclusiveResolved0b3j.selName,MuSelObjAk4JetsLooseExclusiveResolved0b3j.sel))

                        ChannelDictList.append({'channel':'El','sel':ElSelObjAk4JetsLooseExclusiveResolved0b3j.sel,'lep':ElColl[0],'met':self.corrMET,
                                                'j1':self.ak4LightJetsByPt[0],'j2':self.ak4LightJetsByPt[1],'j3':self.ak4LightJetsByPt[2],'j4':None,
                                                'nJet':3,'nbJet':0,
                                                'suffix':ElSelObjAk4JetsLooseExclusiveResolved0b3j.selName,
                                                'is_MC':self.is_MC})
                        ChannelDictList.append({'channel':'Mu','sel':MuSelObjAk4JetsLooseExclusiveResolved0b3j.sel,'lep':MuColl[0],'met':self.corrMET,
                                                'j1':self.ak4LightJetsByPt[0],'j2':self.ak4LightJetsByPt[1],'j3':self.ak4LightJetsByPt[2],'j4':None,
                                                'nJet':3,'nbJet':0,
                                                'suffix':MuSelObjAk4JetsLooseExclusiveResolved0b3j.selName,
                                                'is_MC':self.is_MC})


                #----  Resolved selection : 1 Btag  -----#
                if "LooseResolved1b2j" in jetsel_level:
                    print ("......... Processing Resolved jet (1 btag i.e. bTaggedJets = 1 & nLightJets = 2) selection")
                    ElSelObjAk4JetsLooseExclusiveResolved1b2j = makeExclusiveLooseResolvedJetComboSelection(self,ElSelObjAk4JetsLoose,nbJet=1,copy_sel=True,plot_yield=True)
                    MuSelObjAk4JetsLooseExclusiveResolved1b2j = makeExclusiveLooseResolvedJetComboSelection(self,MuSelObjAk4JetsLoose,nbJet=1,copy_sel=True,plot_yield=True)

                    if not self.args.OnlyYield and "LooseResolved1b2j" in jetplot_level:
                        # Cut flow report #
                        cutFlowPlots.append(CutFlowReport(ElSelObjAk4JetsLooseExclusiveResolved1b2j.selName,ElSelObjAk4JetsLooseExclusiveResolved1b2j.sel))
                        cutFlowPlots.append(CutFlowReport(MuSelObjAk4JetsLooseExclusiveResolved1b2j.selName,MuSelObjAk4JetsLooseExclusiveResolved1b2j.sel))

                        ChannelDictList.append({'channel':'El','sel':ElSelObjAk4JetsLooseExclusiveResolved1b2j.sel,'lep':ElColl[0],'met':self.corrMET,
                                                'j1':self.ak4BJets[0],'j2':self.ak4LightJetsByBtagScore[0],'j3':self.remainingJets[0],'j4':None,
                                                'nJet':3,'nbJet':1,
                                                'suffix':ElSelObjAk4JetsLooseExclusiveResolved1b2j.selName,
                                                'is_MC':self.is_MC})
                        ChannelDictList.append({'channel':'Mu','sel':MuSelObjAk4JetsLooseExclusiveResolved1b2j.sel,'lep':MuColl[0],'met':self.corrMET,
                                                'j1':self.ak4BJets[0],'j2':self.ak4LightJetsByBtagScore[0],'j3':self.remainingJets[0],'j4':None,
                                                'nJet':3,'nbJet':1,
                                                'suffix':MuSelObjAk4JetsLooseExclusiveResolved1b2j.selName,
                                                'is_MC':self.is_MC})

                #----- Resolved selection : 2 Btags -----#
                if "LooseResolved2b1j" in jetsel_level:
                    print ("......... Processing Resolved jet (2 btag i.e. bTaggedJets = 2 & nLightJets = 1) selection")
                    ElSelObjAk4JetsLooseExclusiveResolved2b1j = makeExclusiveLooseResolvedJetComboSelection(self,ElSelObjAk4JetsLoose,nbJet=2,copy_sel=True,plot_yield=True)
                    MuSelObjAk4JetsLooseExclusiveResolved2b1j = makeExclusiveLooseResolvedJetComboSelection(self,MuSelObjAk4JetsLoose,nbJet=2,copy_sel=True,plot_yield=True)

                    if not self.args.OnlyYield and "LooseResolved2b1j" in jetplot_level:
                        # Cut flow report #
                        cutFlowPlots.append(CutFlowReport(ElSelObjAk4JetsLooseExclusiveResolved2b1j.selName,ElSelObjAk4JetsLooseExclusiveResolved2b1j.sel))
                        cutFlowPlots.append(CutFlowReport(MuSelObjAk4JetsLooseExclusiveResolved2b1j.selName,MuSelObjAk4JetsLooseExclusiveResolved2b1j.sel))

                        ChannelDictList.append({'channel':'El',
                                                'sel':ElSelObjAk4JetsLooseExclusiveResolved2b1j.sel,'lep':ElColl[0],'met':self.corrMET,
                                                'j1':self.ak4BJets[0],'j2':self.ak4BJets[1],'j3':self.ak4LightJetsByPt[0],'j4':None,
                                                'nJet':3,'nbJet':2,
                                                'suffix':ElSelObjAk4JetsLooseExclusiveResolved2b1j.selName,
                                                'is_MC':self.is_MC})
                        ChannelDictList.append({'channel':'Mu','sel':MuSelObjAk4JetsLooseExclusiveResolved2b1j.sel,'lep':MuColl[0],'met':self.corrMET,
                                                'j1':self.ak4BJets[0],'j2':self.ak4BJets[1],'j3':self.ak4LightJetsByPt[0],'j4':None,
                                                'nJet':3,'nbJet':2,
                                                'suffix':MuSelObjAk4JetsLooseExclusiveResolved2b1j.selName,
                                                'is_MC':self.is_MC})

                #------------------ Resolved selection (Tight) ---------------------#
                #----- Resolved selection : 0 Btag -----#
                if "TightResolved0b4j" in jetsel_level:
                    print ("......... Processing Tight Resolved jet (0 btag i.e. bTaggedJets = 0 & nLightJets >= 4) selection")
                    ElSelObjAk4JetsTightExclusiveResolved0b4j = makeExclusiveTightResolvedJetComboSelection(self,ElSelObjAk4JetsTight,nbJet=0,copy_sel=True,plot_yield=True)
                    MuSelObjAk4JetsTightExclusiveResolved0b4j = makeExclusiveTightResolvedJetComboSelection(self,MuSelObjAk4JetsTight,nbJet=0,copy_sel=True,plot_yield=True)

                    if not self.args.OnlyYield and "TightResolved0b4j" in jetplot_level:
                        ChannelDictList.append({'channel':'El','sel':ElSelObjAk4JetsTightExclusiveResolved0b4j.sel,'lep':ElColl[0],'met':self.corrMET,
                                                'j1':self.ak4LightJetsByPt[0],'j2':self.ak4LightJetsByPt[1],'j3':self.ak4LightJetsByPt[2],'j4':self.ak4LightJetsByPt[3],
                                                'nJet':4,'nbJet':0,
                                                'suffix':ElSelObjAk4JetsTightExclusiveResolved0b4j.selName,
                                                'is_MC':self.is_MC})
                        ChannelDictList.append({'channel':'Mu','sel':MuSelObjAk4JetsTightExclusiveResolved0b4j.sel,'lep':MuColl[0],'met':self.corrMET,
                                                'j1':self.ak4LightJetsByPt[0],'j2':self.ak4LightJetsByPt[1],'j3':self.ak4LightJetsByPt[2],'j4':self.ak4LightJetsByPt[3],
                                                'nJet':4,'nbJet':0,
                                                'suffix':MuSelObjAk4JetsTightExclusiveResolved0b4j.selName,
                                                'is_MC':self.is_MC})


                #----  Resolved selection : 1 Btag  -----#
                if "TightResolved1b3j" in jetsel_level:
                    print ("......... Processing Resolved jet (1 btag i.e. bTaggedJets = 1 & nLightJets >= 3) selection")
                    ElSelObjAk4JetsTightExclusiveResolved1b3j = makeExclusiveTightResolvedJetComboSelection(self,ElSelObjAk4JetsTight,nbJet=1,copy_sel=True,plot_yield=True)
                    MuSelObjAk4JetsTightExclusiveResolved1b3j = makeExclusiveTightResolvedJetComboSelection(self,MuSelObjAk4JetsTight,nbJet=1,copy_sel=True,plot_yield=True)

                    if not self.args.OnlyYield and "TightResolved1b3j" in jetplot_level:

                        cutFlowPlots.append(CutFlowReport(ElSelObjAk4JetsTightExclusiveResolved1b3j.selName,ElSelObjAk4JetsTightExclusiveResolved1b3j.sel))
                        cutFlowPlots.append(CutFlowReport(MuSelObjAk4JetsTightExclusiveResolved1b3j.selName,MuSelObjAk4JetsTightExclusiveResolved1b3j.sel))

                        if self.args.WhadTagger == 'BDT':

                            self.remainingJetPairsByBDTScore_el_1b3j = op.sort(self.remainingJetPairs(self.remainingJets), lambda jetPair : -self.lambda_evaluateWhadBDT_el(jetPair))
                            self.remainingJetPairsByBDTScore_mu_1b3j = op.sort(self.remainingJetPairs(self.remainingJets), lambda jetPair : -self.lambda_evaluateWhadBDT_mu(jetPair))

                            ChannelDictList.append({'channel':'El','sel':ElSelObjAk4JetsTightExclusiveResolved1b3j.sel,'lep':ElColl[0],'met':self.corrMET,
                                                    'j1':self.ak4BJets[0],'j2':self.ak4LightJetsByBtagScore[0],
                                                    'j3':self.remainingJetPairsByBDTScore_el_1b3j[0][0],'j4':self.remainingJetPairsByBDTScore_el_1b3j[0][1],
                                                    'nJet':4,'nbJet':1,
                                                    'suffix':ElSelObjAk4JetsTightExclusiveResolved1b3j.selName+'_Whad_BDT',
                                                    'is_MC':self.is_MC})
                            ChannelDictList.append({'channel':'Mu','sel':MuSelObjAk4JetsTightExclusiveResolved1b3j.sel,'lep':MuColl[0],'met':self.corrMET,
                                                    'j1':self.ak4BJets[0],'j2':self.ak4LightJetsByBtagScore[0],
                                                    'j3':self.remainingJetPairsByBDTScore_mu_1b3j[0][0],'j4':self.remainingJetPairsByBDTScore_mu_1b3j[0][1],
                                                    'nJet':4,'nbJet':1,
                                                    'suffix':MuSelObjAk4JetsTightExclusiveResolved1b3j.selName+'_Whad_BDT',
                                                    'is_MC':self.is_MC})

                        elif self.args.WhadTagger == 'Simple': 
                            # Simple reco method to find the jet paits coming from W
                            # https://gitlab.cern.ch/cms-hh-bbww/cms-hh-to-bbww/-/blob/master/Legacy/signal_extraction.md
                            self.lambda_chooseWjj_mu1b3j  = lambda dijet: op.abs((dijet[0].p4+dijet[1].p4+MuColl[0].p4+self.corrMET.p4).M() - 
                                                                                 (self.HLL.bJetCorrP4(self.ak4BJets[0]) + self.HLL.bJetCorrP4(self.ak4LightJetsByBtagScore[0])).M()) 
                            self.WjjPairs_mu1b3j = op.sort(self.remainingJetPairs(self.remainingJets), self.lambda_chooseWjj_mu1b3j)
                            
                            self.lambda_chooseWjj_el1b3j  = lambda dijet: op.abs((dijet[0].p4+dijet[1].p4+ElColl[0].p4+self.corrMET.p4).M() - 
                                                                             (self.HLL.bJetCorrP4(self.ak4BJets[0]) + self.HLL.bJetCorrP4(self.ak4LightJetsByBtagScore[0])).M()) 
                            self.WjjPairs_el1b3j = op.sort(self.remainingJetPairs(self.remainingJets), self.lambda_chooseWjj_el1b3j)
                            
                            ChannelDictList.append({'channel':'El','sel':ElSelObjAk4JetsTightExclusiveResolved1b3j.sel,'lep':ElColl[0],'met':self.corrMET,
                                                    'j1':self.ak4BJets[0],'j2':self.ak4LightJetsByBtagScore[0],'j3':self.WjjPairs_el1b3j[0][0],'j4':self.WjjPairs_el1b3j[0][1],
                                                    'nJet':4,'nbJet':1,
                                                    'suffix':ElSelObjAk4JetsTightExclusiveResolved1b3j.selName+'_Whad_Simple',
                                                    'is_MC':self.is_MC})
                            ChannelDictList.append({'channel':'Mu','sel':MuSelObjAk4JetsTightExclusiveResolved1b3j.sel,'lep':MuColl[0],'met':self.corrMET,
                                                    'j1':self.ak4BJets[0],'j2':self.ak4LightJetsByBtagScore[0],'j3':self.WjjPairs_mu1b3j[0][0],'j4':self.WjjPairs_mu1b3j[0][1],
                                                    'nJet':4,'nbJet':1,
                                                    'suffix':MuSelObjAk4JetsTightExclusiveResolved1b3j.selName+'_Whad_Simple',
                                                    'is_MC':self.is_MC})
                        else :
                            raise RuntimeError("Type of WhadTagger not mentioned for TightResolved1b3j category : No Plots")
                            

                #----- Resolved selection : 2 Btags -----#
                if "TightResolved2b2j" in jetsel_level:
                    print ("......... Processing Resolved jet (2 btag i.e. bTaggedJets >= 2 & nLightJets >= 2) selection")    
                    ElSelObjAk4JetsTightExclusiveResolved2b2j = makeExclusiveTightResolvedJetComboSelection(self,ElSelObjAk4JetsTight,nbJet=2,copy_sel=True,plot_yield=True)
                    MuSelObjAk4JetsTightExclusiveResolved2b2j = makeExclusiveTightResolvedJetComboSelection(self,MuSelObjAk4JetsTight,nbJet=2,copy_sel=True,plot_yield=True)

                    if not self.args.OnlyYield and "TightResolved2b2j" in jetplot_level:
                        # Cut flow report #
                        cutFlowPlots.append(CutFlowReport(ElSelObjAk4JetsTightExclusiveResolved2b2j.selName,ElSelObjAk4JetsTightExclusiveResolved2b2j.sel))
                        cutFlowPlots.append(CutFlowReport(MuSelObjAk4JetsTightExclusiveResolved2b2j.selName,MuSelObjAk4JetsTightExclusiveResolved2b2j.sel))

                        if self.args.WhadTagger == 'BDT':

                            self.remainingJetPairsByBDTScore_el_2b2j = op.sort(self.remainingJetPairs(self.ak4LightJetsByPt), lambda jetPair : -self.lambda_evaluateWhadBDT_el(jetPair))
                            self.remainingJetPairsByBDTScore_mu_2b2j = op.sort(self.remainingJetPairs(self.ak4LightJetsByPt), lambda jetPair : -self.lambda_evaluateWhadBDT_mu(jetPair))

                            ChannelDictList.append({'channel':'El','sel':ElSelObjAk4JetsTightExclusiveResolved2b2j.sel,'lep':ElColl[0],'met':self.corrMET,
                                                    'j1':self.ak4BJets[0],'j2':self.ak4BJets[1],
                                                    'j3':self.remainingJetPairsByBDTScore_el_2b2j[0][0],'j4':self.remainingJetPairsByBDTScore_el_2b2j[0][1],
                                                    'nJet':4,'nbJet':2,
                                                    'suffix':ElSelObjAk4JetsTightExclusiveResolved2b2j.selName+'_Whad_BDT',
                                                    'is_MC':self.is_MC})
                            ChannelDictList.append({'channel':'Mu','sel':MuSelObjAk4JetsTightExclusiveResolved2b2j.sel,'lep':MuColl[0],'met':self.corrMET,
                                                    'j1':self.ak4BJets[0],'j2':self.ak4BJets[1],
                                                    'j3':self.remainingJetPairsByBDTScore_mu_2b2j[0][0],'j4':self.remainingJetPairsByBDTScore_mu_2b2j[0][1],
                                                    'nJet':4,'nbJet':2,
                                                    'suffix':MuSelObjAk4JetsTightExclusiveResolved2b2j.selName+'_Whad_BDT',
                                                    'is_MC':self.is_MC})


                        elif self.args.WhadTagger == 'Simple':
                            # Simple reco method to find the jet paits coming from W
                            # https://gitlab.cern.ch/cms-hh-bbww/cms-hh-to-bbww/-/blob/master/Legacy/signal_extraction.md
                            
                            self.lambda_chooseWjj_mu2b2j  = lambda dijet: op.abs((dijet[0].p4+dijet[1].p4+MuColl[0].p4+self.corrMET.p4).M() - 
                                                                                 (self.HLL.bJetCorrP4(self.ak4BJets[0]) + self.HLL.bJetCorrP4(self.ak4LightJetsByBtagScore[0])).M()) 
                            self.WjjPairs_mu2b2j          = op.sort(self.remainingJetPairs(self.ak4LightJetsByPt), self.lambda_chooseWjj_mu2b2j)
                            
                            self.lambda_chooseWjj_el2b2j  = lambda dijet: op.abs((dijet[0].p4+dijet[1].p4+ElColl[0].p4+self.corrMET.p4).M() - 
                                                                                 (self.HLL.bJetCorrP4(self.ak4BJets[0]) + self.HLL.bJetCorrP4(self.ak4LightJetsByBtagScore[0])).M())
                            self.WjjPairs_el2b2j          = op.sort(self.remainingJetPairs(self.ak4LightJetsByPt), self.lambda_chooseWjj_el2b2j)
                            
                            
                            ChannelDictList.append({'channel':'El','sel':ElSelObjAk4JetsTightExclusiveResolved2b2j.sel,'lep':ElColl[0],'met':self.corrMET,
                                                    'j1':self.ak4BJets[0],'j2':self.ak4BJets[1],'j3':self.WjjPairs_el2b2j[0][0],'j4':self.WjjPairs_el2b2j[0][1],
                                                    'nJet':4,'nbJet':2,
                                                    'suffix':ElSelObjAk4JetsTightExclusiveResolved2b2j.selName+'_Whad_Simple',
                                                    'is_MC':self.is_MC})
                            ChannelDictList.append({'channel':'Mu','sel':MuSelObjAk4JetsTightExclusiveResolved2b2j.sel,'lep':MuColl[0],'met':self.corrMET,
                                                    'j1':self.ak4BJets[0],'j2':self.ak4BJets[1],'j3':self.WjjPairs_mu2b2j[0][0],'j4':self.WjjPairs_mu2b2j[0][1],
                                                    'nJet':4,'nbJet':2,
                                                    'suffix':MuSelObjAk4JetsTightExclusiveResolved2b2j.selName+'_Whad_Simple',
                                                    'is_MC':self.is_MC})

                        else :
                            raise RuntimeError("Type of WhadTagger not mentioned for TightResolved2b2j category : No Plots")

                # Lepton + jet Plots #
                ResolvedBTaggedJetsN = {'objName':'Ak4BJets','objCont':self.ak4BJets,'Nmax':5,'xTitle':'N(Ak4 Bjets)'}
                ResolvedLightJetsN   = {'objName':'Ak4LightJets','objCont':self.ak4LightJetsByBtagScore,'Nmax':10,'xTitle':'N(Ak4 Lightjets)'}
        
                for channelDict in ChannelDictList:
                    # Dilepton #
                    plots.extend(makeSinleptonPlots(**{k:channelDict[k] for k in LeptonKeys}))
                    # Number of jets #
                    plots.append(objectsNumberPlot(**{k:channelDict[k] for k in commonItems},**ResolvedBTaggedJetsN))
                    plots.append(objectsNumberPlot(**{k:channelDict[k] for k in commonItems},**ResolvedLightJetsN))
                    # Ak4 Jets #
                    plots.extend(makeAk4JetsPlots(**{k:channelDict[k] for k in JetKeys},HLL=self.HLL))
                    # MET #
                    plots.extend(makeMETPlots(**{k:channelDict[k] for k in commonItems}, met=self.corrMET))
                    # High level #
                    plots.extend(makeHighLevelPlotsResolved(**{k:channelDict[k] for k in ResolvedKeys},HLL=self.HLL))

            if any("SemiBoosted" in key for key in jetsel_level):
                print ("......... processing Semi-Boosted Category")
                ChannelDictList   = []
                FatJetKeys     = ['channel','sel','j1','j2','j3','has1fat1slim','has1fat2slim','suffix']
                FatJetsN       = {'objName':'Ak8Jets','objCont':self.ak8Jets,'Nmax':5,'xTitle':'N(Ak8 jets)'}
                FatBJetsN      = {'objName':'Ak8BJets','objCont':self.ak8BJets,'Nmax':5,'xTitle':'N(Ak8 b-jets)'}
                BoostedKeys    = ['channel','sel','met','lep','j1','j2','j3','suffix','bothAreFat','has1fat2slim']
                if "SemiBoostedHbbWtoJ" in jetsel_level:
                    print ("............ Processing Semi-Boosted category (Htobb:Ak8 + WtoJ)")
                    ElSelObjAk8bJetsHbbBoostedWtoJ = makeSemiBoostedHbbSelection(self,ElSelObjAk8bJets,nNonb=1,copy_sel=True,plot_yield=True)
                    MuSelObjAk8bJetsHbbBoostedWtoJ = makeSemiBoostedHbbSelection(self,MuSelObjAk8bJets,nNonb=1,copy_sel=True,plot_yield=True)
                    if not self.args.OnlyYield and "SemiBoostedHbbWtoJ" in jetplot_level:
                        # Cut flow report #
                        cutFlowPlots.append(CutFlowReport(ElSelObjAk8bJetsHbbBoostedWtoJ.selName,ElSelObjAk8bJetsHbbBoostedWtoJ.sel))
                        cutFlowPlots.append(CutFlowReport(MuSelObjAk8bJetsHbbBoostedWtoJ.selName,MuSelObjAk8bJetsHbbBoostedWtoJ.sel))
 
                        ChannelDictList.append({'channel':'El','sel':ElSelObjAk8bJetsHbbBoostedWtoJ.sel,'lep':ElColl[0],'met':self.corrMET,
                                                'j1':self.ak8BJets[0],'j2':self.ak4JetsCleanedFromAk8b[0],'j3':None,
                                                'has1fat1slim':True,'has1fat2slim':False,'bothAreFat':False,
                                                'suffix':ElSelObjAk8bJetsHbbBoostedWtoJ.selName,
                                                'is_MC':self.is_MC})
                        ChannelDictList.append({'channel':'Mu','sel':MuSelObjAk8bJetsHbbBoostedWtoJ.sel,'lep':MuColl[0],'met':self.corrMET,
                                                'j1':self.ak8BJets[0],'j2':self.ak4JetsCleanedFromAk8b[0],'j3':None,
                                                'has1fat1slim':True,'has1fat2slim':False,'bothAreFat':False,
                                                'suffix':MuSelObjAk8bJetsHbbBoostedWtoJ.selName,
                                                'is_MC':self.is_MC})

                if "SemiBoostedHbbWtoJJ" in jetsel_level:
                    print ("............ Processing Semi-Boosted category (Htobb:Ak8 + WtoJJ)")
                    ElSelObjAk8bJetsHbbBoostedWtoJJ = makeSemiBoostedHbbSelection(self,ElSelObjAk8bJets,nNonb=2,copy_sel=True,plot_yield=True)
                    MuSelObjAk8bJetsHbbBoostedWtoJJ = makeSemiBoostedHbbSelection(self,MuSelObjAk8bJets,nNonb=2,copy_sel=True,plot_yield=True)
                    if not self.args.OnlyYield and "SemiBoostedHbbWtoJJ" in jetplot_level:

                        if self.args.WhadTagger == 'BDT':

                            self.remainingJetPairsByBDTScore_el_Hbb = op.sort(self.remainingJetPairs(self.ak4JetsCleanedFromAk8b), lambda jetPair : -self.lambda_evaluateWhadBDT_el(jetPair))
                            self.remainingJetPairsByBDTScore_mu_Hbb = op.sort(self.remainingJetPairs(self.ak4JetsCleanedFromAk8b), lambda jetPair : -self.lambda_evaluateWhadBDT_mu(jetPair))

                            ChannelDictList.append({'channel':'El','sel':ElSelObjAk8bJetsHbbBoostedWtoJJ.sel,'lep':ElColl[0],'met':self.corrMET,
                                                    'j1':self.ak8BJets[0],'j2':self.remainingJetPairsByBDTScore_el_Hbb[0][0],'j3':self.remainingJetPairsByBDTScore_el_Hbb[0][1],
                                                    'has1fat1slim':False,'has1fat2slim':True,'bothAreFat':False,
                                                    'suffix':ElSelObjAk8bJetsHbbBoostedWtoJJ.selName+'_Whad_BDT',
                                                    'is_MC':self.is_MC})
                            ChannelDictList.append({'channel':'Mu','sel':MuSelObjAk8bJetsHbbBoostedWtoJJ.sel,'lep':MuColl[0],'met':self.corrMET,
                                                    'j1':self.ak8BJets[0],'j2':self.remainingJetPairsByBDTScore_mu_Hbb[0][0],'j3':self.remainingJetPairsByBDTScore_mu_Hbb[0][1],
                                                    'has1fat1slim':False,'has1fat2slim':True,'bothAreFat':False,
                                                    'suffix':MuSelObjAk8bJetsHbbBoostedWtoJJ.selName+'_Whad_BDT',
                                                    'is_MC':self.is_MC})

                        elif self.args.WhadTagger == 'Simple':

                            self.lambda_chooseWjj_muSB    = lambda dijet: op.abs((dijet[0].p4+dijet[1].p4+MuColl[0].p4+self.corrMET.p4).M() - self.ak8BJets[0].p4.M()) 
                            self.WjjPairs_muSB            = op.sort(self.remainingJetPairs(self.ak4JetsCleanedFromAk8b), self.lambda_chooseWjj_muSB)
                            self.lambda_chooseWjj_elSB    = lambda dijet: op.abs((dijet[0].p4+dijet[1].p4+ElColl[0].p4+self.corrMET.p4).M() - self.ak8BJets[0].p4.M()) 
                            #self.nonbJetPairs_elSB        = op.combine(self.ak4JetsCleanedFromAk8b, N=2)
                            self.WjjPairs_elSB            = op.sort(self.remainingJetPairs(self.ak4JetsCleanedFromAk8b), self.lambda_chooseWjj_elSB)
                            
                            # cut flow report
                            cutFlowPlots.append(CutFlowReport(ElSelObjAk8bJetsHbbBoostedWtoJJ.selName,ElSelObjAk8bJetsHbbBoostedWtoJJ.sel))
                            cutFlowPlots.append(CutFlowReport(MuSelObjAk8bJetsHbbBoostedWtoJJ.selName,MuSelObjAk8bJetsHbbBoostedWtoJJ.sel))
                            
                            ChannelDictList.append({'channel':'El','sel':ElSelObjAk8bJetsHbbBoostedWtoJJ.sel,'lep':ElColl[0],'met':self.corrMET,
                                                    'j1':self.ak8BJets[0],'j2':self.WjjPairs_elSB[0][0],'j3':self.WjjPairs_elSB[0][1],
                                                    'has1fat1slim':False,'has1fat2slim':True,'bothAreFat':False,
                                                    'suffix':ElSelObjAk8bJetsHbbBoostedWtoJJ.selName+'_Whad_Simple',
                                                    'is_MC':self.is_MC})
                            ChannelDictList.append({'channel':'Mu','sel':MuSelObjAk8bJetsHbbBoostedWtoJJ.sel,'lep':MuColl[0],'met':self.corrMET,
                                                    'j1':self.ak8BJets[0],'j2':self.WjjPairs_muSB[0][0],'j3':self.WjjPairs_muSB[0][1],
                                                    'has1fat1slim':False,'has1fat2slim':True,'bothAreFat':False,
                                                    'suffix':MuSelObjAk8bJetsHbbBoostedWtoJJ.selName+'_Whad_Simple',
                                                    'is_MC':self.is_MC})
                        else :
                            raise RuntimeError("Type of WhadTagger not mentioned for SemiBoostedHbb category : No Plots")


                for channelDict in ChannelDictList:
                    # Dilepton #
                    plots.extend(makeSinleptonPlots(**{k:channelDict[k] for k in LeptonKeys}))
                    # Number of jets #
                    plots.append(objectsNumberPlot(**{k:channelDict[k] for k in commonItems},**FatJetsN))
                    plots.append(objectsNumberPlot(**{k:channelDict[k] for k in commonItems},**FatBJetsN))       
                    # Ak8 Jets #
                    plots.extend(makeSingleLeptonAk8JetsPlots(**{k:channelDict[k] for k in FatJetKeys}, nMedBJets=self.nMediumBTaggedSubJets))
                    # MET #
                    plots.extend(makeMETPlots(**{k:channelDict[k] for k in commonItems}, met=self.corrMET))
                    # HighLevel #
                    plots.extend(makeHighLevelPlotsBoosted(**{k:channelDict[k] for k in BoostedKeys}, HLL=self.HLL))
            
            if "Boosted" in jetsel_level:
                print ("......... Processing Boosted category selection")        
                ChannelDictList= []
                FatJetKeys     = ['channel','sel','j1','j2','has2fat','suffix']
                FatJetsN       = {'objName':'Ak8Jets','objCont':self.ak8Jets,'Nmax':5,'xTitle':'N(Ak8 jets)'}
                FatBJetsN      = {'objName':'Ak8BJets','objCont':self.ak8BJets,'Nmax':5,'xTitle':'N(Ak8 b-jets)'}
                ElSelObjAk8JetsInclusiveBoosted = makeInclusiveBoostedSelection(self,ElSelObjAk8Jets,copy_sel=True,plot_yield=True)
                MuSelObjAk8JetsInclusiveBoosted = makeInclusiveBoostedSelection(self,MuSelObjAk8Jets,copy_sel=True,plot_yield=True)
                if not self.args.OnlyYield and "Boosted" in jetplot_level:
                    # Cut flow report #
                    cutFlowlots.append(CutFlowReport(ElSelObjAk8JetsInclusiveBoosted.selName,ElSelObjAk8JetsInclusiveBoosted.sel))
                    cutFlowlots.append(CutFlowReport(MuSelObjAk8JetsInclusiveBoosted.selName,MuSelObjAk8JetsInclusiveBoosted.sel))
                    
                    ChannelDictList.append({'channel':'El','sel':ElSelObjAk8JetsInclusiveBoosted.sel,'sinlepton':ElColl[0],
                                            'j1':self.ak8BJets[0],'j2':self.ak8nonBJets[0],'has2fat':True,
                                            'suffix':ElSelObjAk8JetsInclusiveBoosted.selName,
                                            'is_MC':self.is_MC})
                    ChannelDictList.append({'channel':'Mu','sel':MuSelObjAk8JetsInclusiveBoosted.sel,'sinlepton':MuColl[0],
                                            'j1':self.ak8Jets[0],'j2':self.ak8nonBJets[0],'has2fat':True,
                                            'suffix':MuSelObjAk8JetsInclusiveBoosted.selName,
                                            'is_MC':self.is_MC})
                    
                for channelDict in ChannelDictList:
                    # Dilepton #
                    plots.extend(makeSinleptonPlots(**{k:channelDict[k] for k in LeptonKeys}))
                    # Number of jets #
                    plots.append(objectsNumberPlot(**{k:channelDict[k] for k in commonItems},**FatJetsN))
                    plots.append(objectsNumberPlot(**{k:channelDict[k] for k in commonItems},**FatBJetsN))       
                    # Ak8 Jets #
                    plots.extend(makeSingleLeptonAk8JetsPlots(**{k:channelDict[k] for k in FatJetKeys},nMedBJets=self.nMediumBTaggedSubJets))
                    # MET #
                    plots.extend(makeMETPlots(**{k:channelDict[k] for k in commonItems}, met=self.corrMET))



            #----- Machine Learning plots -----#
            if 'BDT' in self.args.Classifier:
                channelDictListFullR = []
                channelDictListMissR = []
                channelDictListFullB = []
                channelDictListMissB = []
                if not self.args.OnlyYield:
                    MLplotkeys=['channel','sel','suffix','nBins']
                    # --------------------------------------------------------- MISS RECO --------------------------------------------------------------- #
                    if self.args.LooseResolved1b2j or self.args.LooseResolved2b1j or self.args.SemiBoostedHbbWtoJ:
                        if self.args.Classifier == 'BDT-SM':
                            modelOdd_  = BDT_missRecoSM_simple_odd
                            modelEven_ = BDT_missRecoSM_simple_even
                        if self.args.Classifier == 'BDT-Rad900':
                            modelOdd_  = BDT_missReco900R_simple_odd
                            modelEven_ = BDT_missReco900R_simple_even

                    bla = 100
                    print(type(bla))
                    if "LooseResolved1b2j" in jetplot_level:
                        channelDictListMissR.append({'channel':'El','fakeLepColl':self.electronsFakeSel,'lep':ElColl[0],'bJets':self.ak4BJets,
                                                     'j1':self.ak4BJets[0],'j2':self.ak4LightJetsByBtagScore[0],'j3':self.remainingJets[0],'j4':None,
                                                     'sel':ElSelObjAk4JetsLooseExclusiveResolved1b2j.sel,
                                                     'suffix':ElSelObjAk4JetsLooseExclusiveResolved1b2j.selName+'_'+self.args.Classifier,'nBins':25})
                        channelDictListMissR.append({'channel':'Mu','fakeLepColl':self.muonsFakeSel,'lep':MuColl[0],'bJets':self.ak4BJets,
                                                     'j1':self.ak4BJets[0],'j2':self.ak4LightJetsByBtagScore[0],'j3':self.remainingJets[0],'j4':None,
                                                     'sel':MuSelObjAk4JetsLooseExclusiveResolved1b2j.sel,
                                                     'suffix':MuSelObjAk4JetsLooseExclusiveResolved2b1j.selName+'_'+self.args.Classifier,'nBins':25})
                    
                    if "LooseResolved2b1j" in jetplot_level:
                        channelDictListMissR.append({'channel':'El','fakeLepColl':self.electronsFakeSel,'lep':ElColl[0],'bJets':self.ak4BJets,
                                                     'j1':self.ak4BJets[0],'j2':self.ak4BJets[1],'j3':self.ak4LightJetsByPt[0],'j4':None,
                                                     'sel':ElSelObjAk4JetsLooseExclusiveResolved2b1j.sel,
                                                     'suffix':ElSelObjAk4JetsLooseExclusiveResolved2b1j.selName+'_'+self.args.Classifier,'nBins':25})
                        channelDictListMissR.append({'channel':'Mu','fakeLepColl':self.muonsFakeSel,'lep':MuColl[0],'bJets':self.ak4BJets,
                                                     'j1':self.ak4BJets[0],'j2':self.ak4BJets[1],'j3':self.ak4LightJetsByPt[0],'j4':None,
                                                     'sel':MuSelObjAk4JetsLooseExclusiveResolved2b1j.sel,
                                                     'suffix':MuSelObjAk4JetsLooseExclusiveResolved2b1j.selName+'_'+self.args.Classifier,'nBins':25})
                    
                    if "SemiBoostedHbbWtoJ" in jetplot_level:
                        channelDictListMissB.append({'channel':'El','fakeLepColl':self.electronsFakeSel,'lep':ElColl[0],'bJets':self.ak8BJets,
                                                     'j1':self.ak8BJets[0].subJet1,'j2':self.ak8BJets[0].subJet2,'j3':self.ak4JetsCleanedFromAk8b[0],'j4':None,
                                                     'sel':ElSelObjAk8bJetsHbbBoostedWtoJ.sel,
                                                     'suffix':ElSelObjAk8bJetsHbbBoostedWtoJ.selName+'_'+self.args.Classifier,'nBins':35})
                        
                        channelDictListMissB.append({'channel':'Mu','fakeLepColl':self.muonsFakeSel,'lep':MuColl[0],'bJets':self.ak8BJets,
                                                     'j1':self.ak8BJets[0].subJet1,'j2':self.ak8BJets[0].subJet2,'j3':self.ak4JetsCleanedFromAk8b[0],'j4':None,
                                                     'sel':MuSelObjAk8bJetsHbbBoostedWtoJ.sel,
                                                     'suffix':MuSelObjAk8bJetsHbbBoostedWtoJ.selName+'_'+self.args.Classifier,'nBins':25})
                        
                    for channelDict in channelDictListMissR:
                        BDTOutputMissR = evaluateBDTmissRecoResolved(channelDict['fakeLepColl'], channelDict['lep'], self.corrMET, self.ak4Jets, channelDict['bJets'], 
                                                                     self.ak4BJets[0], self.ak4LightJetsByBtagScore[0], self.remainingJets[0], None, modelEven_, modelOdd_,
                                                                     t.event, self.HLL)
                        plots.extend(plotSingleLeptonBDTResponse(**{k:channelDict[k] for k in MLplotkeys}, output=BDTOutputMissR))

                    for channelDict in channelDictListMissB:
                        BDTOutputMissB = evaluateBDTmissRecoBoosted(channelDict['fakeLepColl'], channelDict['lep'], self.corrMET, self.ak4Jets, channelDict['bJets'], 
                                                                    self.ak4BJets[0].subJet1, self.ak4BJets[0].subJet2, self.ak4JetsCleanedFromAk8b[0], None, modelEven_, modelOdd_,
                                                                    t.event, self.HLL)
                        plots.extend(plotSingleLeptonBDTResponse(**{k:channelDict[k] for k in MLplotkeys}, output=BDTOutputMissB))
                        
                    # ----------------------------------------------- Tight Resolved 2b2j ---------------------------------------------------- #
                    if self.args.TightResolved1b3j or self.args.TightResolved2b2j or self.args.SemiBoostedHbbWtoJJ:
                        if self.args.Classifier == 'BDT-SM':
                            modelOdd_  = BDT_fullRecoSM_simple_odd if self.args.WhadTagger == 'Simple' else BDT_fullRecoSM_BDT_odd
                            modelEven_ = BDT_fullRecoSM_simple_even if self.args.WhadTagger == 'Simple' else BDT_fullRecoSM_BDT_even
                        if self.args.Classifier == 'BDT-Rad900':
                            modelOdd_  = BDT_fullReco900R_simple_odd if self.args.WhadTagger == 'Simple' else BDT_fullReco900R_BDT_odd
                            modelEven_ = BDT_fullReco900R_simple_even if self.args.WhadTagger == 'Simple' else BDT_fullReco900R_BDT_even

                    if "TightResolved2b2j" in jetplot_level:
                        if self.args.WhadTagger == 'Simple':
                            channelDictListFullR.append({'channel':'El','fakeLepColl':self.electronsFakeSel,'lep':ElColl[0],'bJets':self.ak4BJets,
                                                         'j1':self.ak4BJets[0],'j2':self.ak4BJets[1],'j3':self.WjjPairs_el2b2j[0][0],'j4':self.WjjPairs_el2b2j[0][1],
                                                         'sel':ElSelObjAk4JetsTightExclusiveResolved2b2j.sel,
                                                         'suffix':ElSelObjAk4JetsTightExclusiveResolved2b2j.selName+'_Whad_Simple_'+self.args.Classifier,'nBins':35})
                        
                            channelDictListFullR.append({'channel':'Mu','fakeLepColl':self.muonsFakeSel,'lep':MuColl[0],'bJets':self.ak4BJets,
                                                         'j1':self.ak4BJets[0],'j2':self.ak4BJets[1],'j3':self.WjjPairs_mu2b2j[0][0],'j4':self.WjjPairs_mu2b2j[0][1],
                                                         'sel':MuSelObjAk4JetsTightExclusiveResolved2b2j.sel,
                                                         'suffix':MuSelObjAk4JetsTightExclusiveResolved2b2j.selName+'_Whad_Simple_'+self.args.Classifier,'nBins':40})
                        
                        elif self.args.WhadTagger == 'BDT':
                            channelDictListFullR.append({'channel':'El','fakeLepColl':self.electronsFakeSel,'lep':ElColl[0],'bJets':self.ak4BJets,
                                                         'j1':self.ak4BJets[0],'j2':self.ak4BJets[1],'j3':self.remainingJetPairsByBDTScore_el_2b2j[0][0],
                                                         'j4':self.remainingJetPairsByBDTScore_el_2b2j[0][1],
                                                         'sel':ElSelObjAk4JetsTightExclusiveResolved2b2j.sel,
                                                         'suffix':ElSelObjAk4JetsTightExclusiveResolved2b2j.selName+'_Whad_BDT_'+self.args.Classifier,'nBins':35})
                            channelDictListFullR.append({'channel':'Mu','fakeLepColl':self.muonsFakeSel,'lep':MuColl[0],'bJets':self.ak4BJets,
                                                         'j1':self.ak4BJets[0],'j2':self.ak4BJets[1],'j3':self.remainingJetPairsByBDTScore_mu_2b2j[0][0],
                                                         'j4':self.remainingJetPairsByBDTScore_mu_2b2j[0][1],
                                                         'sel':MuSelObjAk4JetsTightExclusiveResolved2b2j.sel,
                                                         'suffix':MuSelObjAk4JetsTightExclusiveResolved2b2j.selName+'_Whad_BDT_'+self.args.Classifier,'nBins':40})
                        else :
                            print("Type of WhadTagger not mentioned for TightResolved2b2j category : No BDT response Plots")

                        
                    # ----------------------------------------------- Tight Resolved 1b3j ---------------------------------------------------- #
                    if "TightResolved1b3j" in jetplot_level:
                        if self.args.WhadTagger == 'Simple':
                            channelDictListFullR.append({'channel':'El','fakeLepColl':self.electronsFakeSel,'lep':ElColl[0],'bJets':self.ak4BJets,
                                                         'j1':self.ak4BJets[0],'j2':self.ak4LightJetsByBtagScore[0],'j3':self.WjjPairs_el1b3j[0][0],'j4':self.WjjPairs_el1b3j[0][1],
                                                         'sel':ElSelObjAk4JetsTightExclusiveResolved1b3j.sel,
                                                         'suffix':ElSelObjAk4JetsTightExclusiveResolved1b3j.selName+'_Whad_Simple_'+self.args.Classifier,'nBins':60})
                            channelDictListFullR.append({'channel':'Mu','fakeLepColl':self.muonsFakeSel,'lep':MuColl[0],'bJets':self.ak4BJets,
                                                         'j1':self.ak4BJets[0],'j2':self.ak4LightJetsByBtagScore[0],'j3':self.WjjPairs_mu1b3j[0][0],'j4':self.WjjPairs_mu1b3j[0][1],
                                                         'sel':MuSelObjAk4JetsTightExclusiveResolved1b3j.sel,
                                                         'suffix':MuSelObjAk4JetsTightExclusiveResolved1b3j.selName+'_Whad_Simple_'+self.args.Classifier,'nBins':75})
                        
                        elif self.args.WhadTagger == 'BDT':
                            channelDictListFullR.append({'channel':'El','fakeLepColl':self.electronsFakeSel,'lep':ElColl[0],'bJets':self.ak4BJets,
                                                         'j1':self.ak4BJets[0],'j2':self.ak4LightJetsByBtagScore[0],
                                                         'j3':self.remainingJetPairsByBDTScore_el_1b3j[0][0],'j4':self.remainingJetPairsByBDTScore_el_1b3j[0][1],
                                                         'sel':ElSelObjAk4JetsTightExclusiveResolved1b3j.sel,
                                                         'suffix':ElSelObjAk4JetsTightExclusiveResolved1b3j.selName+'_Whad_BDT_'+self.args.Classifier,'nBins':60})
                            channelDictListFullR.append({'channel':'Mu','fakeLepColl':self.muonsFakeSel,'lep':MuColl[0],'bJets':self.ak4BJets,
                                                         'j1':self.ak4BJets[0],'j2':self.ak4LightJetsByBtagScore[0],
                                                         'j3':self.remainingJetPairsByBDTScore_mu_1b3j[0][0],'j4':self.remainingJetPairsByBDTScore_mu_1b3j[0][1],
                                                         'sel':MuSelObjAk4JetsTightExclusiveResolved1b3j.sel,
                                                         'suffix':MuSelObjAk4JetsTightExclusiveResolved1b3j.selName+'_Whad_BDT_'+self.args.Classifier,'nBins':75})
                        else :
                            print("Type of WhadTagger not mentioned for TightResolved1b3j category : No BDT response Plots")
                            
                            
                    # ----------------------------------------------- Hbb Boosted Wjj Resolved ---------------------------------------------------- #
                    if "SemiBoostedHbbWtoJJ" in jetplot_level:
                        if self.args.WhadTagger == 'Simple':
                            channelDictListFullB.append({'channel':'El','fakeLepColl':self.electronsFakeSel,'lep':ElColl[0],'bJets':self.ak8BJets,
                                                         'j1':self.ak8BJets[0].subJet1,'j2':self.ak8BJets[0].subJet2,'j3':self.WjjPairs_elSB[0][0],'j4':self.WjjPairs_elSB[0][1],
                                                         'sel':ElSelObjAk8bJetsHbbBoostedWtoJJ.sel,
                                                         'suffix':ElSelObjAk8bJetsHbbBoostedWtoJJ.selName+'_Whad_Simple_'+self.args.Classifier,'nBins':35})
                            channelDictListFullB.append({'channel':'Mu','fakeLepColl':self.muonsFakeSel,'lep':MuColl[0],'bJets':self.ak8BJets,
                                                         'j1':self.ak8BJets[0].subJet1,'j2':self.ak8BJets[0].subJet2,
                                                         'j3':self.WjjPairs_muSB[0][0],'j4':self.WjjPairs_muSB[0][1],
                                                         'sel':MuSelObjAk8bJetsHbbBoostedWtoJJ.sel,
                                                         'suffix':MuSelObjAk8bJetsHbbBoostedWtoJJ.selName+'_Whad_Simple_'+self.args.Classifier,'nBins':50})
                            
                        elif self.args.WhadTagger == 'BDT':
                            channelDictListFullB.append({'channel':'El','fakeLepColl':self.electronsFakeSel,'lep':ElColl[0],'bJets':self.ak8BJets,
                                                         'j1':self.ak8BJets[0].subJet1,'j2':self.ak8BJets[0].subJet2,
                                                         'j3':self.remainingJetPairsByBDTScore_el_Hbb[0][0],'j4':self.remainingJetPairsByBDTScore_el_Hbb[0][1],
                                                         'sel':ElSelObjAk8bJetsHbbBoostedWtoJJ.sel,
                                                         'suffix':ElSelObjAk8bJetsHbbBoostedWtoJJ.selName+'_Whad_BDT_'+self.args.Classifier,'nBins':35})
                            channelDictListFullB.append({'channel':'Mu','fakeLepColl':self.muonsFakeSel,'lep':MuColl[0],'bJets':self.ak8BJets,
                                                         'j1':self.ak8BJets[0].subJet1,'j2':self.ak8BJets[0].subJet2,
                                                         'j3':self.remainingJetPairsByBDTScore_mu_Hbb[0][0],'j4':self.remainingJetPairsByBDTScore_mu_Hbb[0][1],
                                                         'sel':MuSelObjAk8bJetsHbbBoostedWtoJJ.sel,
                                                         'suffix':MuSelObjAk8bJetsHbbBoostedWtoJJ.selName+'_Whad_BDT_'+self.args.Classifier,'nBins':50})
                        else:
                            print('no Whad type or Classifier are mentioned :: No BDT response is plotted! [SemiBoostedHbbWtoJJ]')
                        

                    for channelDict in channelDictListFullR:
                        BDTOutputFullR = evaluateBDTfullRecoResolved(channelDict['fakeLepColl'], channelDict['lep'], self.corrMET, self.ak4Jets, channelDict['bJets'], 
                                                                     channelDict['j1'], channelDict['j2'], channelDict['j3'], channelDict['j4'], modelEven_, modelOdd_,
                                                                     t.event, self.HLL)
                        #print(type(BDTOutputFullR))
                        #print(type(op.static_cast("float",BDTOutputFullR[0])))
                        plots.extend(plotSingleLeptonBDTResponse(**{k:channelDict[k] for k in MLplotkeys}, output=BDTOutputFullR))
                        
                    for channelDict in channelDictListFullB:
                        BDTOutputFullB = evaluateBDTfullRecoBoosted(channelDict['fakeLepColl'], channelDict['lep'], self.corrMET, self.ak4Jets, channelDict['bJets'], 
                                                                    channelDict['j1'], channelDict['j2'], channelDict['j3'], channelDict['j4'], modelEven_, modelOdd_,
                                                                    t.event, self.HLL)
                        plots.extend(plotSingleLeptonBDTResponse(**{k:channelDict[k] for k in MLplotkeys}, output=BDTOutputFullB))


            # ------------------------------------ DNN plots ----------------------------------- #
            if self.args.Classifier == 'DNN':
                selObjectDictList = []
                if "TightResolved2b2j" in jetplot_level:
                    selObjectDictList.append({'channel':'El','lepton':ElColl[0],'selObject':ElSelObjAk4JetsTightExclusiveResolved2b2j,
                                              'fakelepcoll':self.electronsFakeSel,'j3':self.WjjPairs_el2b2j[0][0],'j4':self.WjjPairs_el2b2j[0][1]})
                    selObjectDictList.append({'channel':'Mu','lepton':MuColl[0],'selObject':MuSelObjAk4JetsTightExclusiveResolved2b2j,
                                              'fakelepcoll':self.muonsFakeSel,'j3':self.WjjPairs_mu2b2j[0][0],'j4':self.WjjPairs_mu2b2j[0][1]})
                if "TightResolved1b3j" in jetplot_level:
                    selObjectDictList.append({'channel':'El','lepton':ElColl[0],'selObject':ElSelObjAk4JetsTightExclusiveResolved1b3j,
                                              'fakelepcoll':self.electronsFakeSel,'j3':self.WjjPairs_el1b3j[0][0],'j4':self.WjjPairs_el1b3j[0][1]})
                    selObjectDictList.append({'channel':'Mu','lepton':MuColl[0],'selObject':MuSelObjAk4JetsTightExclusiveResolved1b3j,
                                              'fakelepcoll':self.muonsFakeSel,'j3':self.WjjPairs_mu1b3j[0][0],'j4':self.WjjPairs_mu1b3j[0][1]})
                    
                self.nodes = ['DY', 'H', 'HH', 'Rare', 'ST', 'TT', 'TTVX', 'VVV','WJets']
                
                for selObjDict in selObjectDictList:
                    output = evaluateDNNfullRecoResolved(selObjDict['lepton'],selObjDict['fakelepcoll'],self.corrMET,self.ak4Jets,self.ak4BJets,selObjDict['j3'],selObjDict['j4'],
                                                         DNN_fullRecoResolved_SM_model_01,DNN_fullRecoResolved_SM_model_02,DNN_fullRecoResolved_SM_model_03,
                                                         DNN_fullRecoResolved_SM_model_04,DNN_fullRecoResolved_SM_model_05,t.event,self.HLL)
                    selObjNodesDict = makeDNNOutputNodesSelections(self,selObjDict['selObject'],output,plot_yield=True)
                    for selObjNode in selObjNodesDict.values():
                        cutFlowPlots.append(CutFlowReport(selObjNode.selName,selObjNode.sel))
                    if not self.args.OnlyYield:
                        plots.extend(plotSingleLeptonDNNResponse(selObjNodesDict,output,self.nodes,channel=selObjDict['channel']))
                                
        #----- Add the Yield plots -----#
        plots.extend(self.yieldPlots.returnPlots())
        #plots.extend(cutFlowPlots)
        return plots

