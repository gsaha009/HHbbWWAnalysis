import os
import sys
import json
from itertools import chain

from bamboo import treefunctions as op
from bamboo.analysismodules import NanoAODModule, NanoAODHistoModule, NanoAODSkimmerModule
from bamboo.analysisutils import makeMultiPrimaryDatasetTriggerSelection
from bamboo.scalefactors import binningVariables_nano, BtagSF
from bamboo.plots import SelectionWithDataDriven,CutFlowReport

from METScripts import METFilter, METcorrection
from scalefactorsbbWW import ScaleFactorsbbWW
from btagHelper import makeBtagRatioReweighting
from triggers import returnTriggerRanges
from highlevelLambdas import highlevelLambdas
from DDHelper import DataDrivenPseudoData

#===============================================================================================#
#                                  BaseHHtobbWW                                                 #
#===============================================================================================#
class BaseNanoHHtobbWW(NanoAODModule):
    """ Base module: HH->bbW(->e/µ nu)W(->e/µ nu) histograms from NanoAOD """
    def __init__(self, args):
            super(BaseNanoHHtobbWW, self).__init__(args)
            # Set plots options #
            self.plotDefaults = {"show-ratio": True,
                                 "y-axis-show-zero" : True,
                                 #"normalized": True,
                                 "y-axis": "Events",
                                 "log-y"  : "both",
                                 "ratio-y-axis-range" : [0.8,1.2],
                                 "ratio-y-axis" : '#frac{Data}/{MC}',
                                 "sort-by-yields" : True}

    #-------------------------------------------------------------------------------------------#
    #                                       addArgs                                             #
    #-------------------------------------------------------------------------------------------#
    def addArgs(self,parser):
        super(BaseNanoHHtobbWW, self).addArgs(parser)

        parser.title = """

Arguments for the HH->bbWW analysis on bamboo framework

----- Argument groups -----
    * Lepton arguments *
        --NoZVeto
        --ZPeak
        --NoTauVeto

    * Jet arguments *
        --Ak4
        --Ak8
        --Resolved0Btag
        --Resolved1Btag
        --Resolved2Btag
        --TightResolved0b4j
        --TightResolved1b3j
        --TightResolved2b2j
        --SemiBoostedHbb
        --SemiBoostedWjj
        --TTBarCR
        --BtagReweightingOn
        --BtagReweightingOff

    * Skimmer arguments *
        --Synchronization
        --Channel

    * Plotter arguments *
        --OnlyYield

    * Technical *
        --backend
        --NoSystematics

----- Plotter mode -----
Every combination of lepton and jet arguments can be used, if none specified they will all be plotted

----- Skimmer mode -----
One lepton and and one jet argument must be specified in addition to the required channel
(this is because only one selection at a time can produce a ntuple)

----- Detailed explanation -----

                    """
        #----- Technical arguments -----#
        parser.add_argument("--backend", 
                            type=str, 
                            default="dataframe", 
                            help="Backend to use, 'dataframe' (default) or 'lazy'")
        parser.add_argument("--NoSystematics", 
                            action      = "store_true",
                            default     = False,
                            help="Disable all systematic variations (default=False)")
        parser.add_argument("--Events", 
                            nargs       = '+',
                            type        = int,
                            help="Cut on events (as list)")


        #----- Lepton selection arguments -----#
        parser.add_argument("--POGID", 
                            action      = "store_true",
                            default     = False,
                            help        = "Use the POG id for tight leptons")
        parser.add_argument("--TTHIDLoose", 
                            action      = "store_true",
                            default     = False,
                            help        = "Use the Loose mva ttH ID tight leptons")
        parser.add_argument("--TTHIDTight", 
                            action      = "store_true",
                            default     = False,
                            help        = "Use the Tight mva ttH ID tight leptons")
        parser.add_argument("--NoZVeto", 
                            action      = "store_true",
                            default     = False,
                            help        = "Remove the cut of preselected leptons |M_ll-M_Z|>10 GeV")
        parser.add_argument("--ZPeak", 
                            action      = "store_true",
                            default     = False,
                            help        = "Select the Z peak at tight level |M_ll-M_Z|<10 GeV (must be used with --NoZVeto, only effective with --Tight)")
        parser.add_argument("--NoTauVeto", 
                            action      = "store_true",
                            default     = False,
                            help        = "Select the events do not have any tau overlapped with fakeable leptons")

        #----- Jet selection arguments -----#
        parser.add_argument("--Ak4", 
                                action      = "store_true",
                                default     = False,
                                help        = "Produce the plots/skim for two Ak4 jets passing the selection criteria")
        parser.add_argument("--Ak8", 
                                action      = "store_true",
                                default     = False,
                                help        = "Produce the plots/skim for one Ak8 jet passing the selection criteria")
        parser.add_argument("--Resolved0Btag", 
                                action      = "store_true",
                                default     = False,
                                help        = "Produce the plots/skim for the exclusive resolved category with no btagged jet")
        parser.add_argument("--Resolved1Btag", 
                                action      = "store_true",
                                default     = False,
                                help        = "Produce the plots/skim for the exclusive resolved category with only one btagged jet")
        parser.add_argument("--Resolved2Btag", 
                                action      = "store_true",
                                default     = False,
                                help        = "Produce the plots/skim for the exclusive resolved category with two btagged jets")

        # SL Categories
        # Resolved
        parser.add_argument("--Res2b2Wj", 
                                action      = "store_true",
                                default     = False,
                                help        = "Produce the plots/skim for the exclusive 2b2Wj JPA category")
        parser.add_argument("--Res2b1Wj", 
                                action      = "store_true",
                                default     = False,
                                help        = "Produce the plots/skim for the exclusive 2b1Wj JPA category")
        parser.add_argument("--Res2b0Wj", 
                                action      = "store_true",
                                default     = False,
                                help        = "Produce the plots/skim for the exclusive 2b0Wj JPA category")
        parser.add_argument("--Res1b2Wj", 
                                action      = "store_true",
                                default     = False,
                                help        = "Produce the plots/skim for the exclusive 1b2Wj JPA category")
        parser.add_argument("--Res1b1Wj", 
                                action      = "store_true",
                                default     = False,
                                help        = "Produce the plots/skim for the exclusive 1b1Wj JPA category")
        parser.add_argument("--Res1b0Wj", 
                                action      = "store_true",
                                default     = False,
                                help        = "Produce the plots/skim for the exclusive 1b0Wj JPA category")
        parser.add_argument("--Res0b", 
                                action      = "store_true",
                                default     = False,
                                help        = "Produce the plots/skim for the exclusive 0b JPA category")

        # Semi-Boosted
        parser.add_argument("--Hbb2Wj", 
                                action      = "store_true",
                                default     = False,
                                help        = "Produce the plots/skim for the exclusive Hbb2Wj JPA category")
        parser.add_argument("--Hbb1Wj", 
                                action      = "store_true",
                                default     = False,
                                help        = "Produce the plots/skim for the exclusive Hbb1Wj JPA category")
        parser.add_argument("--Hbb0Wj", 
                                action      = "store_true",
                                default     = False,
                                help        = "Produce the plots/skim for the exclusive Hbb0Wj JPA category")

        parser.add_argument("--Resolved", 
                                action      = "store_true",
                                default     = False,
                                help        = "Produce the plots/skim for all JPA resolved categories")
        parser.add_argument("--Boosted", 
                                action      = "store_true",
                                default     = False,
                                help        = "Produce the plots/skim for all JPA boosted categories")
        ################

        parser.add_argument("--Boosted0Btag", 
                                action      = "store_true",
                                default     = False,
                                help        = "Produce the plots/skim for the inclusive boosted category")
        parser.add_argument("--Boosted1Btag", 
                                action      = "store_true",
                                default     = False,
                                help        = "Produce the plots/skim for the inclusive boosted category")
        parser.add_argument("--TTBarCR", 
                                action      = "store_true",
                                default     = False,
                                help        = "Apply cut on Mbb for ttbar CR (only effective with --Resolved2Btag)")
        parser.add_argument("--BtagReweightingOn", 
                                action      = "store_true",
                                default     = False,
                                help        = "Btag ratio study : Btag SF applied (without the ratio), will only do the plots for reweighting (jets and leptons args are ignored)")
        parser.add_argument("--BtagReweightingOff", 
                                action      = "store_true",
                                default     = False,
                                help        = "Btag ratio study : Btag Sf not applied (without the ratio), will only do the plots for reweighting (jets and leptons args are ignored)")
        parser.add_argument("--DYStitchingPlots", 
                                action      = "store_true",
                                default     = False,
                                help        = "DY stitching studies : only produce LHE jet multiplicities (inclusive analysis, only on DY events, rest of plots ignored)")
        parser.add_argument("--WJetsStitchingPlots", 
                                action      = "store_true",
                                default     = False,
                                help        = "W+jets stitching studies : only produce LHE jet multiplicities (inclusive analysis, only on W+jets events, rest of plots ignored)")
        parser.add_argument("--NoStitching", 
                                action      = "store_true",
                                default     = False,
                                help        = "To not apply the stitching weights to DY and WJets samples")
        parser.add_argument("--DYVariable", 
                                action      = "store",
                                default     = False,
                                help        = "DY data estimation variable")

        #----- Skimmer arguments -----#
        parser.add_argument("--Synchronization", 
                            action      = "store_true",
                            default     = False,
                            help        = "Produce the skims for the synchronization (without triggers, corrections of flags) if alone. If sync for specific selection, lepton, jet and channel arguments need to be used")
        parser.add_argument("--Channel", 
                            action      = "store",
                            type        = str,
                            help        = "Specify the channel for the tuple : ElEl, MuMu, ElMu")
        parser.add_argument("--FakeCR", 
                            action      = "store_true",
                            default     = False,
                            help        = "Use the Fake CR instead of the SR")

        #----- Plotter arguments -----#
        parser.add_argument("--OnlyYield", 
                            action      = "store_true",
                            default     = False,
                            help        = "Only produce the yield plots")
        parser.add_argument("--Classifier", 
                            action      = "store",
                            type        = str,
                            help        = "BDT-SM | BDT-Rad900 | DNN | LBN")
        parser.add_argument("--WhadTagger", 
                            action      = "store",
                            type        = str,
                            help        = "BDT | simple")


    def prepareTree(self, tree, sample=None, sampleCfg=None):
        from bamboo.treedecorators import NanoAODDescription, nanoRochesterCalc, nanoJetMETCalc, nanoJetMETCalc_METFixEE2017, nanoFatJetCalc
        # JEC's Recommendation for Full RunII: https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC
        # JER : -----------------------------: https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution

        # Get base aguments #
        era = sampleCfg['era']
        self.is_MC = self.isMC(sample) # no confusion between boolean is_MC and method isMC()
        metName = "METFixEE2017" if era == "2017" else "MET"
        tree,noSel,be,lumiArgs = super(BaseNanoHHtobbWW,self).prepareTree(tree          = tree, 
                                                                          sample        = sample, 
                                                                          sampleCfg     = sampleCfg, 
                                                                          description   = NanoAODDescription.get(
                                                                                            tag             = "v7", 
                                                                                            year            = (era if era else "2016"),
                                                                                            isMC            = self.is_MC,
                                                                                            systVariations  = [ (nanoJetMETCalc_METFixEE2017 if era == "2017" else nanoJetMETCalc), nanoFatJetCalc]),
                                                                                            # will do Jet and MET variations, and not the Rochester correction
                                                                          lazyBackend   = (self.args.backend == "lazy" or self.args.onlypost))
    
        self.triggersPerPrimaryDataset = {}
        from bamboo.analysisutils import configureJets ,configureRochesterCorrection, configureType1MET 

        # Event cut #
        if self.args.Events:
            print ("Events to use only :")
            for e in self.args.Events:
                print ('... %d'%e)
            noSel = noSel.refine('eventcut',cut = [op.OR(*[tree.event == e for e in self.args.Events])])

        # Save some useful stuff in self #
        self.sample = sample
        self.sampleCfg = sampleCfg
        self.era = era

        # Check if v7#
        if self.is_MC:
            self.isNanov7 = ('db' in sampleCfg.keys() and 'NanoAODv7' in sampleCfg['db']) or ('files' in sampleCfg.keys() and all(['NanoAODv7' in f for f in sampleCfg['files']]))
        else:
            self.isNanov7 = ('db' in sampleCfg.keys() and '02Apr2020' in sampleCfg['db']) or ('files' in sampleCfg.keys() and all(['02Apr2020' in f for f in sampleCfg['files']]))
        if self.isNanov7:
            print ("Using NanoAODv7")

        # Check distributed option #
        isNotWorker = (self.args.distributed != "worker") 

        # Turn off systs #
        if self.args.NoSystematics:
            noSel = noSel.refine('SystOff',autoSyst=False)

        # Check era #
        if era != "2016" and era != "2017" and era != "2018":
            raise RuntimeError("Unknown era {0}".format(era))

        # Rochester and JEC corrections (depends on era) #     

        # Check if basic synchronization is required (no corrections and triggers) #
        self.inclusive_sel = ((self.args.Synchronization 
                               and not any([self.args.__dict__[key] for key in['Ak4', 'Ak8', 'Resolved0Btag', 'Resolved1Btag', 'Resolved2Btag', 'Boosted0Btag','Boosted1Btag',
                                                                               'Resolved','Boosted','Res2b2Wj','Res2b1Wj','Res2b0Wj','Res1b2Wj','Res1b1Wj','Res1b1Wj','Res0b',
                                                                               'Hbb2Wj','Hbb1Wj','Hbb0Wj']]) \
                               and (self.args.Channel is None or self.args.Channel=='None')) \
                              # No channel selection
                              # None is local mode, "None" in distributed mode
                              or self.args.BtagReweightingOn 
                              or self.args.BtagReweightingOff
                              or self.args.DYStitchingPlots
                              or self.args.WJetsStitchingPlots)
        # Inclusive plots
        # If no lepton, jet and channel selection : basic object selection (no trigger nor corrections)
        if self.inclusive_sel:
            print ("Inclusive analysis, no selections applied")

        #----- Theory uncertainties -----#
        if self.is_MC:
            #qcdScaleVariations = { f"qcdScalevar{i}": tree.LHEScaleWeight[i] for i in [0, 1, 3, 5, 7, 8] }
            #qcdScaleSyst = op.systematic(op.c_float(1.), name="qcdScale", **qcdScaleVariations)
            self.psISRSyst = op.systematic(op.c_float(1.), name="psISR", up=tree.PSWeight[2], down=tree.PSWeight[0])
            self.psFSRSyst = op.systematic(op.c_float(1.), name="psFSR", up=tree.PSWeight[3], down=tree.PSWeight[1])
            #pdfsWeight = op.systematic(op.c_float(1.), name="pdfsWgt", up=tree.LHEPdfWeight, down=tree.LHEPdfWeight)
            #noSel = noSel.refine("theorySystematics", weight = [qcdScaleSyst,psISRSyst,psFSRSyst,pdfsWeight])
            noSel = noSel.refine("PSweights", weight = [self.psISRSyst, self.psFSRSyst])

        #----- Triggers and Corrections -----#
        self.triggersPerPrimaryDataset = {}
        def addHLTPath(key,HLT):
            if key not in self.triggersPerPrimaryDataset.keys():
                self.triggersPerPrimaryDataset[key] = []
            try:
                self.triggersPerPrimaryDataset[key].append(getattr(tree.HLT,HLT))
            except AttributeError:
                print ("Could not find branch tree.HLT.%s, will omit it"%HLT)

        ############################################################################################
        # ERA 2016 #
        ############################################################################################
        if era == "2016":
#            # Rochester corrections #
#            configureRochesterCorrection(variProxy  = tree._Muon,
#                                         paramsFile = os.path.join(os.path.dirname(__file__), "data", "RoccoR2016.txt"),
#                                         isMC       = self.is_MC,
#                                         backend    = be, 
#                                         uName      = sample)
 
            # SingleMuon #
            addHLTPath("SingleMuon","IsoMu22")
            addHLTPath("SingleMuon","IsoTkMu22")
            addHLTPath("SingleMuon","IsoMu22_eta2p1")
            addHLTPath("SingleMuon","IsoTkMu22_eta2p1")
            addHLTPath("SingleMuon","IsoMu24")
            addHLTPath("SingleMuon","IsoTkMu24")
            # SingleElectron #
            addHLTPath("SingleElectron","Ele27_WPTight_Gsf")
            addHLTPath("SingleElectron","Ele25_eta2p1_WPTight_Gsf")
            addHLTPath("SingleElectron","Ele27_eta2p1_WPLoose_Gsf")
            # DoubleMuon #
            addHLTPath("DoubleMuon","Mu17_TrkIsoVVL_Mu8_TrkIsoVVL")
            addHLTPath("DoubleMuon","Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ")
            addHLTPath("DoubleMuon","Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL")
            addHLTPath("DoubleMuon","Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ")
            # DoubleEGamma #
            addHLTPath("DoubleEGamma","Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ")
            # MuonEG #
            addHLTPath("MuonEG","Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL")
            addHLTPath("MuonEG","Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ")
            addHLTPath("MuonEG","Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL")
            addHLTPath("MuonEG","Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_DZ")

            # Links : 
            # JEC : https://twiki.cern.ch/twiki/bin/viewauth/CMS/JECDataMC
            # JER (smear) : https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution
            # JetMET treatment #
            if not self.args.Synchronization:
                if self.is_MC:   # if MC -> needs smearing
                    configureJets(variProxy             = tree._Jet, 
                                  jetType               = "AK4PFchs",
                                  jec                   = "Summer16_07Aug2017_V11_MC",
                                  smear                 = "Summer16_25nsV1_MC",
                                  jesUncertaintySources = "Merged",
                                  regroupTag            = "V2",
                                  mayWriteCache         = isNotWorker,
                                  isMC                  = self.is_MC,
                                  backend               = be, 
                                  uName                 = sample)
                    configureJets(variProxy             = tree._FatJet, 
                                  jetType               = "AK8PFPuppi", 
                                  jec                   = "Summer16_07Aug2017_V11_MC", 
                                  smear                 = "Summer16_25nsV1_MC", 
                                  jesUncertaintySources = ["Total"],
                                  mcYearForFatJets      = era, 
                                  mayWriteCache         = isNotWorker, 
                                  isMC                  = self.is_MC,
                                  backend               = be, 
                                  uName                 = sample)
                    configureType1MET(variProxy             = getattr(tree, f"_{metName}"),
                                      jec                   = "Summer16_07Aug2017_V11_MC",
                                      smear                 = "Summer16_25nsV1_MC",
                                      jesUncertaintySources = "Merged",
                                      regroupTag            = "V2",
                                      mayWriteCache         = isNotWorker,
                                      isMC                  = self.is_MC,
                                      backend               = be,
                                      uName                 = sample)

                else:                   # If data -> extract info from config 
                    jecTag = None
                    if "2016B" in sample or "2016C" in sample or "2016D" in sample:
                        jecTag = "Summer16_07Aug2017BCD_V11_DATA"
                    elif "2016E" in sample or "2016F" in sample:
                        jecTag = "Summer16_07Aug2017EF_V11_DATA"
                    elif "2016G" in sample or "2016H" in sample:
                        jecTag = "Summer16_07Aug2017GH_V11_DATA"
                    else:
                        raise RuntimeError("Could not find appropriate JEC tag for data")
                    configureJets(variProxy             = tree._Jet, 
                                  jetType               = "AK4PFchs",
                                  jec                   = jecTag,
                                  mayWriteCache         = isNotWorker,
                                  isMC                  = self.is_MC,
                                  backend               = be, 
                                  uName                 = sample)
                    configureJets(variProxy             = tree._FatJet, 
                                  jetType               = "AK8PFPuppi", 
                                  jec                   = jecTag,
                                  jesUncertaintySources = ["Total"],
                                  mcYearForFatJets      = era, 
                                  mayWriteCache         = isNotWorker, 
                                  isMC                  = self.is_MC,
                                  backend               = be, 
                                  uName                 = sample)
                    configureType1MET(variProxy         = getattr(tree, f"_{metName}"),
                                      jec               = jecTag,
                                      mayWriteCache     = isNotWorker,
                                      isMC              = self.is_MC,
                                      backend           = be, 
                                      uName             = sample)

        ############################################################################################
        # ERA 2017 #
        ############################################################################################
        elif era == "2017":
            #configureRochesterCorrection(variProxy  = tree._Muon,
            #                             paramsFile = os.path.join(os.path.dirname(__file__), "data", "RoccoR2017.txt"),
            #                             isMC       = self.is_MC,
            #                             backend    = be, 
            #                             uName      = sample)

            # SingleMuon #
            addHLTPath("SingleMuon","IsoMu24")
            addHLTPath("SingleMuon","IsoMu27")
            # SingleElectron #
            addHLTPath("SingleElectron","Ele35_WPTight_Gsf")
            addHLTPath("SingleElectron","Ele32_WPTight_Gsf")
            # DoubleMuon #
            addHLTPath("DoubleMuon","Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8")
            addHLTPath("DoubleMuon","Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8")
            # DoubleEGamma #
            addHLTPath("DoubleEGamma","Ele23_Ele12_CaloIdL_TrackIdL_IsoVL")
            # MuonEG #
            addHLTPath("MuonEG","Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ")
            addHLTPath("MuonEG","Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ")
            addHLTPath("MuonEG","Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL")
            addHLTPath("MuonEG","Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ")
            
            # JetMET treatment #
            if not self.args.Synchronization:
                if self.is_MC:   # if MC -> needs smearing
                    configureJets(variProxy             = tree._Jet, 
                                  jetType               = "AK4PFchs",
                                  jec                   = "Fall17_17Nov2017_V32_MC",
                                  smear                 = "Fall17_V3b_MC",
                                  jesUncertaintySources = "Merged",
                                  regroupTag            = "V2",
                                  mayWriteCache         = isNotWorker,
                                  isMC                  = self.is_MC,
                                  backend               = be, 
                                  uName                 = sample)
                    configureJets(variProxy             = tree._FatJet, 
                                  jetType               = "AK8PFPuppi", 
                                  jec                   = "Fall17_17Nov2017_V32_MC",
                                  smear                 = "Fall17_V3b_MC",
                                  jesUncertaintySources = ["Total"],
                                  mcYearForFatJets      = era, 
                                  mayWriteCache         = isNotWorker, 
                                  isMC                  = self.is_MC,
                                  backend               = be, 
                                  uName                 = sample)
                    configureType1MET(variProxy             = getattr(tree, f"_{metName}"),
                                      jec                   = "Fall17_17Nov2017_V32_MC",
                                      smear                 = "Fall17_V3b_MC",
                                      jesUncertaintySources = "Merged",
                                      regroupTag            = "V2",
                                      mayWriteCache         = isNotWorker,
                                      isMC                  = self.is_MC,
                                      backend               = be,
                                      uName                 = sample)

                else:                   # If data -> extract info from config 
                    jecTag = None
                    if "2017B" in sample:
                        jecTag = "Fall17_17Nov2017B_V32_DATA"
                    elif "2017C" in sample:
                        jecTag = "Fall17_17Nov2017C_V32_DATA"
                    elif "2017D" in sample or "2017E" in sample:
                        jecTag = "Fall17_17Nov2017DE_V32_DATA"
                    elif "2017F" in sample:
                        jecTag = "Fall17_17Nov2017F_V32_DATA"
                    else:
                        raise RuntimeError("Could not find appropriate JEC tag for data")
                    configureJets(variProxy             = tree._Jet, 
                                  jetType               = "AK4PFchs",
                                  jec                   = jecTag,
                                  mayWriteCache         = isNotWorker,
                                  isMC                  = self.is_MC,
                                  backend               = be, 
                                  uName                 = sample)
                    configureJets(variProxy             = tree._FatJet, 
                                  jetType               = "AK8PFPuppi", 
                                  jec                   = jecTag,
                                  jesUncertaintySources = ["Total"],
                                  mcYearForFatJets      = era, 
                                  mayWriteCache         = isNotWorker, 
                                  isMC                  = self.is_MC,
                                  backend               = be, 
                                  uName                 = sample)
                    configureType1MET(variProxy         = getattr(tree, f"_{metName}"),
                                      jec               = jecTag,
                                      mayWriteCache     = isNotWorker,
                                      isMC              = self.is_MC,
                                      backend           = be, 
                                      uName             = sample)

        ############################################################################################
        # ERA 2018 #
        ############################################################################################
        elif era == "2018":
            #configureRochesterCorrection(variProxy  = tree._Muon,
            #                             paramsFile = os.path.join(os.path.dirname(__file__), "data", "RoccoR2018.txt"),
            #                             isMC       = self.is_MC,
            #                             backend    = be, 
            #                             uName      = sample)

            # SingleMuon #
            addHLTPath("SingleMuon","IsoMu24")
            addHLTPath("SingleMuon","IsoMu27")
            # SingleElectron #
            addHLTPath("SingleElectron","Ele32_WPTight_Gsf")
            # DoubleMuon #
            addHLTPath("DoubleMuon","Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8")
            # DoubleEGamma #
            addHLTPath("DoubleEGamma","Ele23_Ele12_CaloIdL_TrackIdL_IsoVL")
            # MuonEG #
            addHLTPath("MuonEG","Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ")
            addHLTPath("MuonEG","Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ")
            addHLTPath("MuonEG","Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL")

            # JetMET treatment #
            if not self.args.Synchronization:
                if self.is_MC:   # if MC -> needs smearing
                    configureJets(variProxy             = tree._Jet, 
                                  jetType               = "AK4PFchs",
                                  jec                   = "Autumn18_V19_MC",
                                  smear                 = "Autumn18_V7b_MC",
                                  jesUncertaintySources = "Merged",
                                  regroupTag            = "V2",
                                  mayWriteCache         = isNotWorker,
                                  isMC                  = self.is_MC,
                                  backend               = be, 
                                  uName                 = sample)
                    configureJets(variProxy             = tree._FatJet, 
                                  jetType               = "AK8PFPuppi", 
                                  jec                   = "Autumn18_V19_MC",
                                  smear                 = "Autumn18_V7b_MC",
                                  jesUncertaintySources = ["Total"],
                                  mcYearForFatJets      = era, 
                                  mayWriteCache         = isNotWorker, 
                                  isMC                  = self.is_MC,
                                  backend               = be, 
                                  uName                 = sample)
                    configureType1MET(variProxy             = getattr(tree, f"_{metName}"),
                                      jec                   = "Autumn18_V19_MC",
                                      smear                 = "Autumn18_V7b_MC",
                                      jesUncertaintySources = "Merged",
                                      regroupTag            = "V2",
                                      mayWriteCache         = isNotWorker,
                                      isMC                  = self.is_MC,
                                      backend               = be,
                                      uName                 = sample)

                else:                   # If data -> extract info from config 
                    jecTag = None
                    if "2018A" in sample:
                        jecTag = "Autumn18_RunA_V19_DATA"
                    elif "2018B" in sample:
                        jecTag = "Autumn18_RunB_V19_DATA"
                    elif "2018C" in sample:
                        jecTag = "Autumn18_RunC_V19_DATA"
                    elif "2018D" in sample:
                        jecTag = "Autumn18_RunD_V19_DATA"
                    else:
                        raise RuntimeError("Could not find appropriate JEC tag for data")
                    configureJets(variProxy             = tree._Jet, 
                                  jetType               = "AK4PFchs",
                                  jec                   = jecTag,
                                  mayWriteCache         = isNotWorker,
                                  isMC                  = self.is_MC,
                                  backend               = be, 
                                  uName                 = sample)
                    configureJets(variProxy             = tree._FatJet, 
                                  jetType               = "AK8PFPuppi", 
                                  jec                   = jecTag,
                                  jesUncertaintySources = ["Total"], 
                                  mcYearForFatJets      = era, 
                                  mayWriteCache         = isNotWorker, 
                                  isMC                  = self.is_MC,
                                  backend               = be, 
                                  uName                 = sample)
                    configureType1MET(variProxy         = getattr(tree, f"_{metName}"),
                                      jec               = jecTag,
                                      mayWriteCache     = isNotWorker,
                                      isMC              = self.is_MC,
                                      backend           = be, 
                                      uName             = sample)

        return tree,noSel,be,lumiArgs

    def initialize(self,forSkimmer=False):
        super(BaseNanoHHtobbWW, self).initialize()
        if not forSkimmer and "PseudoData" in self.datadrivenContributions:
            contrib = self.datadrivenContributions["PseudoData"]
            self.datadrivenContributions["PseudoData"] = DataDrivenPseudoData(contrib.name, contrib.config)

    def prepareObjects(self, t, noSel, sample, sampleCfg, channel, forSkimmer=False):
        # Some imports #

        if channel not in ["DL","SL"]:
            raise RuntimeError('Channel %s not understood'%channel)

        era = sampleCfg['era']
        self.era = era
        self.tree = t

        self.yields = CutFlowReport("yields",printInLog=self.args.Events is not None,recursive=self.args.Events is not None)

        ###########################################################################
        #                              Pseudo-data                                #
        ###########################################################################
        if not forSkimmer and "PseudoData" in self.datadrivenContributions: # Skimmer does not know about self.datadrivenContributions
            noSel = SelectionWithDataDriven.create(parent   = noSel,
                                                   name     = 'pseudodata',
                                                   ddSuffix = 'Pseudodata',
                                                   enable   = "PseudoData" in self.datadrivenContributions 
                                    and self.datadrivenContributions["PseudoData"].usesSample(self.sample, self.sampleCfg))

        ###########################################################################
        #                           TTbar reweighting                             #
        ###########################################################################
        if "group" in sampleCfg and sampleCfg["group"] == 'ttbar': 
            print ('Applied TT top reweighting')
            # https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopPtReweighting#Use_case_3_ttbar_MC_is_used_to_m -> do not use when ttbar is background
            # Correct : https://indico.cern.ch/event/904971/contributions/3857701/attachments/2036949/3410728/TopPt_20.05.12.pdf
            #       -> Weight formula : slide 2
            #       -> top/antitop SF : slide 12 bottom left 
            # Get tops #
            self.genTop = op.select(t.GenPart,lambda g : op.AND(g.pdgId==6, g.statusFlags & ( 0x1 << 13)))
            self.genAntitop = op.select(t.GenPart,lambda g : op.AND(g.pdgId==-6, g.statusFlags & ( 0x1 << 13)))
                # statusFlags==13 : isLastCopy
                # Pdgid == 6 : top
            # Lambda to compute weight if there is a ttbar #
            self.t_SF = lambda t : op.exp(-2.02274e-01 + 1.09734e-04*t.pt + -1.30088e-07*t.pt**2 + (5.83494e+01/(t.pt+1.96252e+02)))
            self.ttbar_weight = lambda t,tbar : op.sqrt(self.t_SF(t)*self.t_SF(tbar))
            self.ttbar_sys = op.systematic(self.ttbar_weight(self.genTop[0],self.genAntitop[0]),
                                           name = "ttbarweightsyst",
                                           up   = op.pow(self.ttbar_weight(self.genTop[0],self.genAntitop[0]),2), # Up = apply twice
                                           down = op.c_float(1))                                                  # Down = not apply
            # Apply correction to TT #
            noSel = noSel.refine("ttbarWeight",weight=self.ttbar_sys)

        ###########################################################################
        #                               Stitching                                 #
        ###########################################################################
        if "group" in sampleCfg and (sampleCfg["group"] == 'DY' or sampleCfg["group"] == 'Wjets') and not self.args.NoStitching:
            stitch_file = os.path.abspath(os.path.join(os.path.dirname(__file__),'data','Stitching','stitching_weights_{}_{}.json'.format(sampleCfg["group"],era)))
            if not os.path.exists(stitch_file):
                raise RuntimeError("Could not find stitching file %s"%stitch_file)
            with open(stitch_file) as handle:
                dict_weights = json.load(handle)
            if sample in dict_weights.keys():
                dict_weights = dict_weights[sample]
                if isinstance(dict_weights[list(dict_weights.keys())[0]],float): # Only binned in jet multiplicity
                    stitch_op = op.multiSwitch(*[(t.LHE.Njets==int(njets),op.c_float(weight)) for njets,weight in dict_weights.items()], op.c_float(1.))
                elif isinstance(dict_weights[list(dict_weights.keys())[0]],list): # Binned in jet mult + HT bins
                    stitch_op = op.multiSwitch(*[(op.AND(t.LHE.Njets==int(njets),op.in_range(weights['low'],t.LHE.HT,weights['up'])),op.c_float(weights['value'])) 
                                                  for njets,listBin in dict_weights.items() for weights in listBin], op.c_float(1.))
                else:
                    raise RuntimeError("Stitching weight format not understood")
                noSel = noSel.refine("DYStitching",weight = stitch_op)
                
        ###########################################################################
        #                               tH samples                                #
        ###########################################################################
        if sample.startswith('TH'):
            # https://twiki.cern.ch/twiki/bin/viewauth/CMS/SingleTopHiggsGeneration13TeV
            # https://twiki.cern.ch/twiki/pub/CMS/SingleTopHiggsGeneration13TeV/reweight_encondig.txt
            # 
            print ('Applied tH LHE weights')
            noSel = noSel.refine("tHWeight",weight=t.LHEReweightingWeight[11])

        #############################################################################
        #                            Pre-firing rates                               #
        #############################################################################
        if era in ["2016","2017"] and self.is_MC:
            self.L1Prefiring = op.systematic(t.L1PreFiringWeight_Nom,
                                             name = "L1PreFiring",
                                             up   = t.L1PreFiringWeight_Up,
                                             down = t.L1PreFiringWeight_Dn)
            
            noSel = noSel.refine("L1PreFiringRate", weight = self.L1Prefiring)

        #############################################################################
        #                             Pile-up                                       #
        #############################################################################
        # Get MC PU weight file #
        puWeightsFile = None
        if self.is_MC:
            puWeightsFile = os.path.join(os.path.dirname(__file__), "data", "pileup",sample+'_%s.json'%era)
            if not os.path.exists(puWeightsFile):
                raise RuntimeError("Could not find pileup file %s"%puWeightsFile)
            from bamboo.analysisutils import makePileupWeight
            PUWeight = makePileupWeight(puWeightsFile, t.Pileup_nTrueInt, systName="pileup",nameHint=f"puweightFromFile{sample}".replace('-','_'))
            noSel = noSel.refine("puWeight", weight = PUWeight)

        #############################################################################
        #                                 MET                                       #
        #############################################################################
        # MET filter #
        if not self.inclusive_sel:
            noSel = noSel.refine("passMETFlags", cut=METFilter(t.Flag, era, self.is_MC) )

        # MET corrections #
        self.rawMET = t.MET if era != "2017" else t.METFixEE2017
        self.corrMET = METcorrection(self.rawMET,t.PV,sample,era,self.is_MC) 


        #############################################################################
        #                      Lepton Lambdas Variables                             #
        #############################################################################
        # Associated jet Btagging #
        self.lambda_hasAssociatedJet = lambda lep : lep.jet.idx != -1
        if era == "2016": 
            self.lambda_lepton_associatedJetLessThanMediumBtag = lambda lep : op.OR(op.NOT(self.lambda_hasAssociatedJet(lep)),
                                                                        lep.jet.btagDeepFlavB <= 0.3093)
            self.lambda_lepton_associatedJetLessThanTightBtag  = lambda lep : op.OR(op.NOT(self.lambda_hasAssociatedJet(lep)),
                                                                        lep.jet.btagDeepFlavB <= 0.7221)
        elif era =="2017":
            self.lambda_lepton_associatedJetLessThanMediumBtag = lambda lep : op.OR(op.NOT(self.lambda_hasAssociatedJet(lep)),
                                                                        lep.jet.btagDeepFlavB <= 0.3033)
            self.lambda_lepton_associatedJetLessThanTightBtag  = lambda lep : op.OR(op.NOT(self.lambda_hasAssociatedJet(lep)),
                                                                        lep.jet.btagDeepFlavB <= 0.7489)
        elif era == "2018":
            self.lambda_lepton_associatedJetLessThanMediumBtag = lambda lep : op.OR(op.NOT(self.lambda_hasAssociatedJet(lep)),
                                                                        lep.jet.btagDeepFlavB <= 0.2770)
            self.lambda_lepton_associatedJetLessThanTightBtag  = lambda lep : op.OR(op.NOT(self.lambda_hasAssociatedJet(lep)),
                                                                        lep.jet.btagDeepFlavB <= 0.7264)
                     # If no associated jet, isolated lepton : cool !  

        # Cone pt #
        if self.args.POGID:
            self.lambda_conept_electron = lambda lep : lep.pt
            self.lambda_conept_muon = lambda lep : lep.pt
        if self.args.TTHIDTight:
            # Def conept : https://github.com/CERN-PH-CMG/cmgtools-lite/blob/f8a34c64a4489d94ff9ac4c0d8b0b06dad46e521/TTHAnalysis/python/tools/conept.py#L74
            self.lambda_conept_electron = lambda lep : op.multiSwitch((op.AND(op.abs(lep.pdgId)!=11 , op.abs(lep.pdgId)!=13) , op.static_cast("Float_t",lep.pt)),
                                                                      # if (abs(lep.pdgId)!=11 and abs(lep.pdgId)!=13): return lep.pt : anything that is not muon or electron
                                                                      (op.AND(op.abs(lep.pdgId)==11 , lep.mvaTTH > 0.80) , op.static_cast("Float_t",lep.pt)),
                                                                      # if electron, check above MVA 
                                                                       op.static_cast("Float_t",0.9*lep.pt*(1.+lep.jetRelIso)))
                                                                      # else: return 0.90 * lep.pt / jetPtRatio where jetPtRatio = 1./(Electron_jetRelIso + 1.)
            self.lambda_conept_muon = lambda lep : op.multiSwitch((op.AND(op.abs(lep.pdgId)!=11 , op.abs(lep.pdgId)!=13) , op.static_cast("Float_t",lep.pt)),
                                                                   # if (abs(lep.pdgId)!=11 and abs(lep.pdgId)!=13): return lep.pt : anything that is not muon or electron
                                                                  (op.AND(op.abs(lep.pdgId)==13 , lep.mediumId ,lep.mvaTTH > 0.85) , op.static_cast("Float_t",lep.pt)),
                                                                   # if muon, check that passes medium and above MVA
                                                                   op.static_cast("Float_t",0.9*lep.pt*(1.+lep.jetRelIso)))
                                                               # else: return 0.90 * lep.pt / lep.jetPtRatiov2
        if self.args.TTHIDLoose:
            self.lambda_conept_electron = lambda lep : op.multiSwitch((op.AND(op.abs(lep.pdgId)!=11 , op.abs(lep.pdgId)!=13) , op.static_cast("Float_t",lep.pt)),
                                                                      # if (abs(lep.pdgId)!=11 and abs(lep.pdgId)!=13): return lep.pt : anything that is not muon or electron
                                                                      (op.AND(op.abs(lep.pdgId)==11 , lep.mvaTTH > 0.30) , op.static_cast("Float_t",lep.pt)),
                                                                      # if electron, check above MVA 
                                                                       op.static_cast("Float_t",0.9*lep.pt*(1.+lep.jetRelIso)))
                                                                      # else: return 0.90 * lep.pt / jetPtRatio where jetPtRatio = 1./(Electron_jetRelIso + 1.)
            self.lambda_conept_muon = lambda lep : op.multiSwitch((op.AND(op.abs(lep.pdgId)!=11 , op.abs(lep.pdgId)!=13) , op.static_cast("Float_t",lep.pt)),
                                                                   # if (abs(lep.pdgId)!=11 and abs(lep.pdgId)!=13): return lep.pt : anything that is not muon or electron
                                                                  (op.AND(op.abs(lep.pdgId)==13 , lep.mediumId ,lep.mvaTTH > 0.50) , op.static_cast("Float_t",lep.pt)),
                                                                   # if muon, check that passes medium and above MVA
                                                                   op.static_cast("Float_t",0.9*lep.pt*(1.+lep.jetRelIso)))
                                                               # else: return 0.90 * lep.pt / lep.jetPtRatiov2

        self.electron_conept = op.map(t.Electron, self.lambda_conept_electron)
        self.muon_conept = op.map(t.Muon, self.lambda_conept_muon)

        # Btag interpolation #
                    # https://indico.cern.ch/event/812025/contributions/3475878/attachments/1867083/3070589/gp-fr-run2b.pdf (slide 7)
        self.lambda_muon_x = lambda mu : op.min(op.max(0.,(0.9*mu.pt*(1+mu.jetRelIso))-20.)/(45.-20.), 1.)
                    # x = min(max(0, jet_pt-PT_min)/(PT_max-PT_min), 1) where jet_pt = 0.9*PT_muon*(1+MuonJetRelIso), PT_min=25, PT_max=40
        if era == "2016": 
            self.lambda_muon_btagInterpolation = lambda mu : self.lambda_muon_x(mu)*0.0614 + (1-self.lambda_muon_x(mu))*0.3093
        elif era =="2017":
            self.lambda_muon_btagInterpolation = lambda mu : self.lambda_muon_x(mu)*0.0521 + (1-self.lambda_muon_x(mu))*0.3033
        elif era == "2018":
            self.lambda_muon_btagInterpolation = lambda mu : self.lambda_muon_x(mu)*0.0494 + (1-self.lambda_muon_x(mu))*0.2770
            # return x*WP_loose+(1-x)*WP_medium
        #self.lambda_muon_deepJetInterpIfMvaFailed = lambda mu : mu.jet.btagDeepFlavB < self.lambda_muon_btagInterpolation(mu)
        self.lambda_muon_deepJetInterpIfMvaFailed = lambda mu : op.OR(op.NOT(self.lambda_hasAssociatedJet(mu)),     # If no associated jet, isolated lepton : cool !
                                                                      mu.jet.btagDeepFlavB < self.lambda_muon_btagInterpolation(mu))

        # Dilepton lambdas #
        self.lambda_leptonOS  = lambda l1,l2 : l1.charge != l2.charge
        if self.is_MC:
            self.lambda_is_matched = lambda lep : op.OR(lep.genPartFlav==1,  # Prompt muon or electron
                                                        lep.genPartFlav==15) # From tau decay
                                                        #lep.genPartFlav==22) # From photon conversion (only available for electrons)
        else:
            self.lambda_is_matched = lambda lep : op.c_bool(True)
        
        # Tight and fake selections 
        # Tight Dilepton : must also be Gen matched if MC #
        self.lambda_dilepton_matched = lambda dilep : op.AND(self.lambda_is_matched(dilep[0]),self.lambda_is_matched(dilep[1]))
        self.lambda_tightpair_ElEl = lambda dilep : op.AND(self.lambda_dilepton_matched(dilep),
                                                           self.lambda_electronTightSel(dilep[0]),
                                                           self.lambda_electronTightSel(dilep[1]))
        self.lambda_tightpair_MuMu = lambda dilep : op.AND(self.lambda_dilepton_matched(dilep),
                                                           self.lambda_muonTightSel(dilep[0]),
                                                           self.lambda_muonTightSel(dilep[1]))
        self.lambda_tightpair_ElMu = lambda dilep : op.AND(self.lambda_dilepton_matched(dilep),
                                                           self.lambda_electronTightSel(dilep[0]),
                                                           self.lambda_muonTightSel(dilep[1]))
             
        # Fake Extrapolation dilepton #               
        self.lambda_fakepair_ElEl = lambda dilep : op.AND(self.lambda_dilepton_matched(dilep),
                                                          op.NOT(op.AND(self.lambda_electronTightSel(dilep[0]),
                                                                        self.lambda_electronTightSel(dilep[1]))))
        self.lambda_fakepair_MuMu = lambda dilep : op.AND(self.lambda_dilepton_matched(dilep),
                                                          op.NOT(op.AND(self.lambda_muonTightSel(dilep[0]),
                                                                        self.lambda_muonTightSel(dilep[1]))))
        self.lambda_fakepair_ElMu = lambda dilep : op.AND(self.lambda_dilepton_matched(dilep),
                                                          op.NOT(op.AND(self.lambda_electronTightSel(dilep[0]),
                                                                        self.lambda_muonTightSel(dilep[1]))))
                

        #############################################################################
        #                                 Muons                                     #
        #############################################################################
        if self.args.POGID:
            muonsByPt = op.sort(t.Muon, lambda mu : -mu.pt) # Ordering done by pt
            self.lambda_muonTightSel = lambda mu : op.AND(mu.tightId,
                                                          mu.pfRelIso04_all <= 0.15,
                                                          mu.pt >= 15.,
                                                          op.abs(mu.eta) <= 2.4)
            self.muonsTightSel = op.select(muonsByPt, self.lambda_muonTightSel)
                                                        
        if self.args.TTHIDLoose or self.args.TTHIDTight:
            muonsByPt = op.sort(t.Muon, lambda mu : -self.muon_conept[mu.idx]) # Ordering done by conept
            # Preselection #
            self.lambda_muonPreSel = lambda mu : op.AND(mu.pt >= 5.,
                                                        op.abs(mu.eta) <= 2.4,
                                                        op.abs(mu.dxy) <= 0.05,
                                                        op.abs(mu.dz) <= 0.1,
                                                        mu.miniPFRelIso_all <= 0.4, # mini PF relative isolation, total (with scaled rho*EA PU corrections)
                                                        mu.sip3d <= 8,
                                                        mu.looseId)

            self.muonsPreSel = op.select(muonsByPt, self.lambda_muonPreSel)
            # Fakeable selection #
            if self.args.TTHIDTight:
                self.lambda_muonFakeSel = lambda mu : op.AND(self.muon_conept[mu.idx] >= op.c_float(10.),
                                                             self.lambda_lepton_associatedJetLessThanMediumBtag(mu),
                                                             op.OR(mu.mvaTTH >= 0.85, op.AND(mu.jetRelIso<0.5 , self.lambda_muon_deepJetInterpIfMvaFailed(mu))))
                                                            # If mvaTTH < 0.85 : jetRelIso <0.5 and < deepJet medium with interpolation
            if self.args.TTHIDLoose:
                self.lambda_muonFakeSel = lambda mu : op.AND(self.muon_conept[mu.idx] >= op.c_float(10.),
                                                             self.lambda_lepton_associatedJetLessThanMediumBtag(mu),
                                                             op.OR(mu.mvaTTH >= 0.50, op.AND(mu.jetRelIso<0.8 , self.lambda_muon_deepJetInterpIfMvaFailed(mu))))
                                                            # If mvaTTH < 0.50 : jetRelIso <0.8 and < deepJet medium with interpolation
            self.muonsFakeSel = op.select(self.muonsPreSel, self.lambda_muonFakeSel)
            # Tight selection #
            if self.args.TTHIDTight:
                self.lambda_muonTightSel = lambda mu : op.AND(mu.mvaTTH >= 0.85, # Lepton MVA id from ttH
                                                              mu.mediumId)
            if self.args.TTHIDLoose:
                self.lambda_muonTightSel = lambda mu : op.AND(mu.mvaTTH >= 0.50, # Lepton MVA id from ttH
                                                              mu.mediumId)

            self.muonsTightSel = op.select(self.muonsFakeSel, self.lambda_muonTightSel)

        #############################################################################
        #                              Electrons                                    #
        #############################################################################
        # Preselection #
        if self.args.POGID:
            electronsByPt = op.sort(t.Electron, lambda ele : -ele.pt) # Ordering done by pt
            self.lambda_electronTightSel = lambda ele : op.AND(ele.mvaFall17V2Iso_WP90,
                                                               ele.pt >= 15.,
                                                               op.abs(ele.eta) <= 2.5)
            self.lambda_cleanElectron = lambda ele : op.NOT(op.rng_any(self.muonsTightSel, lambda mu : op.deltaR(mu.p4, ele.p4) <= 0.3 ))
            self.electronsTightSelBeforeCleaning = op.select(electronsByPt, self.lambda_electronTightSel)
            self.electronsTightSel = op.select(self.electronsTightSelBeforeCleaning, self.lambda_cleanElectron)

        if self.args.TTHIDLoose or self.args.TTHIDTight:
            electronsByPt = op.sort(t.Electron, lambda ele : -self.electron_conept[ele.idx]) # Ordering done by conept
            self.lambda_electronPreSel = lambda ele : op.AND(ele.pt >= 7.,
                                                             op.abs(ele.eta) <= 2.5,
                                                             op.abs(ele.dxy) <= 0.05,
                                                             op.abs(ele.dz) <= 0.1,
                                                             ele.miniPFRelIso_all <= 0.4, # mini PF relative isolation, total (with scaled rho*EA PU corrections)

                                                             ele.sip3d <= 8,
                                                             ele.mvaFall17V2noIso_WPL, 
                                                             ele.lostHits <=1)    # number of missing inner hits

            self.electronsPreSelInclu = op.select(electronsByPt, self.lambda_electronPreSel) # can include a muon in cone
            self.lambda_cleanElectron = lambda ele : op.NOT(op.rng_any(self.muonsPreSel, lambda mu : op.deltaR(mu.p4, ele.p4) <= 0.3 ))
                # No overlap between electron and muon in cone of DR<=0.3
            self.electronsPreSel = op.select(self.electronsPreSelInclu, self.lambda_cleanElectron)
            # Fakeable selection #
            if self.args.TTHIDTight:
                self.lambda_electronFakeSel = lambda ele : op.AND(self.electron_conept[ele.idx] >= 10,
                                                                  op.OR(
                                                                          op.AND(op.abs(ele.eta+ele.deltaEtaSC)<=1.479, ele.sieie<=0.011), 
                                                                          op.AND(op.abs(ele.eta+ele.deltaEtaSC)>1.479, ele.sieie<=0.030)),
                                                                  self.lambda_lepton_associatedJetLessThanMediumBtag(ele),
                                                                  ele.hoe <= 0.10,
                                                                  ele.eInvMinusPInv >= -0.04,
                                                                  op.OR(ele.mvaTTH >= 0.80, op.AND(ele.jetRelIso<0.7, ele.mvaFall17V2noIso_WP80)), # Lepton MVA id from ttH
                                                                      # If mvaTTH < 0.80 : jetRelIso <0.7 and Fall17V2noIso_WP80
                                                                  ele.lostHits == 0,    # number of missing inner hits
                                                                  ele.convVeto)        # Passes conversion veto
            if self.args.TTHIDLoose:
                self.lambda_electronFakeSel = lambda ele : op.AND(self.electron_conept[ele.idx] >= 10,
                                                                  op.OR(
                                                                          op.AND(op.abs(ele.eta+ele.deltaEtaSC)<=1.479, ele.sieie<=0.011), 
                                                                          op.AND(op.abs(ele.eta+ele.deltaEtaSC)>1.479, ele.sieie<=0.030)),
                                                                  ele.hoe <= 0.10,
                                                                  ele.eInvMinusPInv >= -0.04,
                                                                  op.OR(ele.mvaTTH >= 0.30, op.AND(ele.jetRelIso<0.7, ele.mvaFall17V2noIso_WP90)), # Lepton MVA id from ttH
                                                                      # If mvaTTH < 0.30 : jetRelIso <0.7 and Fall17V2noIso_WP90
                                                                  op.switch(ele.mvaTTH < 0.30, 
                                                                            self.lambda_lepton_associatedJetLessThanTightBtag(ele),
                                                                            self.lambda_lepton_associatedJetLessThanMediumBtag(ele)),
                                                                  ele.lostHits == 0,    # number of missing inner hits
                                                                  ele.convVeto)        # Passes conversion veto

            self.electronsFakeSel = op.select(self.electronsPreSel, self.lambda_electronFakeSel)
            # Tight selection #
            if self.args.TTHIDTight:
                self.lambda_electronTightSel = lambda ele : ele.mvaTTH >= 0.80
            if self.args.TTHIDLoose:
                self.lambda_electronTightSel = lambda ele : ele.mvaTTH >= 0.30
            self.electronsTightSel = op.select(self.electronsFakeSel, self.lambda_electronTightSel)

        #############################################################################
        #                               Dileptons                                   #
        #############################################################################

        if self.args.POGID:
            # To remove mll>12 resonances #
            self.ElElTightDileptonPairs = op.combine(self.electronsTightSelBeforeCleaning, N=2)
            self.MuMuTightDileptonPairs = op.combine(self.muonsTightSel, N=2)
            self.ElMuTightDileptonPairs = op.combine((self.electronsTightSelBeforeCleaning,self.muonsTightSel))
            # To remove Z peak resonances #
            self.ElElTightDileptonOSPairs = op.combine(self.electronsTightSel, N=2, pred=lambda l1,l2: l1.charge != l2.charge)
            self.MuMuTightDileptonOSPairs = op.combine(self.muonsTightSel, N=2, pred=lambda l1,l2: l1.charge != l2.charge)

            if channel == "DL":
                self.ElElTightSel = op.combine(self.electronsTightSel, N=2)
                self.MuMuTightSel = op.combine(self.muonsTightSel, N=2)
                self.ElMuTightSel = op.combine((self.electronsTightSel,self.muonsTightSel))

        if self.args.TTHIDLoose or self.args.TTHIDTight:
            #----- Needed for both SL and DL -----#
            # Preselected dilepton -> Mll>12 cut #
            self.ElElDileptonPreSel = op.combine(self.electronsPreSelInclu, N=2)
            self.MuMuDileptonPreSel = op.combine(self.muonsPreSel, N=2)
            self.ElMuDileptonPreSel = op.combine((self.electronsPreSelInclu, self.muonsPreSel))
                # Need pairs without sign for one of the selection
                # We use electrons before muon cleaning : electron-muon pair that has low mass are typically close to each other
            # OS Preselected dilepton -> Z Veto #
            self.OSElElDileptonPreSel = op.combine(self.electronsPreSel, N=2, pred=self.lambda_leptonOS)
            self.OSMuMuDileptonPreSel = op.combine(self.muonsPreSel, N=2, pred=self.lambda_leptonOS)
            self.OSElMuDileptonPreSel = op.combine((self.electronsPreSel, self.muonsPreSel), pred=self.lambda_leptonOS) 

            # Dilepton for selection #
            if channel == "DL":
                self.ElElFakeSel = op.combine(self.electronsFakeSel, N=2)
                self.MuMuFakeSel = op.combine(self.muonsFakeSel, N=2)
                self.ElMuFakeSel = op.combine((self.electronsFakeSel,self.muonsFakeSel))

                self.ElElTightSel = op.combine(self.electronsTightSel, N=2)
                self.MuMuTightSel = op.combine(self.muonsTightSel, N=2)
                self.ElMuTightSel = op.combine((self.electronsTightSel,self.muonsTightSel))

        #############################################################################
        #                               Triggers                                    #
        #############################################################################
        #----- Genweight -----#
        if self.is_MC:
            noSel = noSel.refine("genWeight", weight=t.genWeight)
          
        #----- Trigger mask for data -----#
        if not self.is_MC:
            if era == "2018": # For 2018 the electron samples are merged, need to build according dict for primary dataset
                trigger_per_pd = { pd : trig for pd, trig in self.triggersPerPrimaryDataset.items() if pd not in ("SingleElectron", "DoubleEGamma") } 
                trigger_per_pd["EGamma"] = self.triggersPerPrimaryDataset["SingleElectron"] + self.triggersPerPrimaryDataset["DoubleEGamma"]
                noSel = noSel.refine("DataMaskTriggers", cut=[makeMultiPrimaryDatasetTriggerSelection(sample, trigger_per_pd)])
            else:
                noSel = noSel.refine("DataMaskTriggers", cut=[makeMultiPrimaryDatasetTriggerSelection(sample, self.triggersPerPrimaryDataset)])
            # makeMultiPrimaryDatasetTriggerSelection must be done first, to make sure an event is not passed several times from several datasets
            # Then the trigger selection is based on fakeable leptons

        ##############################################################################
        #                                  Tau                                       #
        ##############################################################################
        self.lambda_tauSel = lambda ta : op.AND(ta.pt > 20.,
                                                op.abs(ta.p4.Eta()) < 2.3,
                                                op.abs(ta.dxy) <= 1000.0,
                                                op.abs(ta.dz) <= 0.2,
                                                ta.idDecayModeNewDMs,
                                                op.OR(ta.decayMode == 0,
                                                      ta.decayMode == 1,
                                                      ta.decayMode == 2,
                                                      ta.decayMode == 10,
                                                      ta.decayMode == 11),
                                                (ta.idDeepTau2017v2p1VSjet >> 4 & 0x1) == 1,
                                                (ta.idDeepTau2017v2p1VSe >> 0 & 0x1)   == 1,
                                                (ta.idDeepTau2017v2p1VSmu >> 0 & 0x1)  == 1
                                            )
        self.tauSel = op.select (t.Tau, self.lambda_tauSel)
        # Cleaning #
        if self.args.POGID:
            self.lambda_tauClean = lambda ta : op.AND(op.NOT(op.rng_any(self.electronsTightSel, lambda el : op.deltaR(ta.p4, el.p4) <= 0.3)), 
                                                      op.NOT(op.rng_any(self.muonsTightSel, lambda mu : op.deltaR(ta.p4, mu.p4) <= 0.3)))
        elif self.args.TTHIDLoose or self.args.TTHIDTight:
            self.lambda_tauClean = lambda ta : op.AND(op.NOT(op.rng_any(self.electronsFakeSel, lambda el : op.deltaR(ta.p4, el.p4) <= 0.3)), 
                                                      op.NOT(op.rng_any(self.muonsFakeSel, lambda mu : op.deltaR(ta.p4, mu.p4) <= 0.3)))
        else:
            self.lambda_tauClean = lambda ta : op.c_float(True)
        self.tauCleanSel = op.select(self.tauSel, self.lambda_tauClean)

        #############################################################################
        #                                AK4 Jets                                   #
        #############################################################################
        self.ak4JetsByPt = op.sort(t.Jet, lambda jet : -jet.pt)
        # Preselection #
        self.lambda_ak4JetsPreSel = lambda j : op.AND(j.jetId & 1 if era == "2016" else j.jetId & 2, # Jet ID flags bit1 is loose, bit2 is tight, bit3 is tightLepVeto
                                                      j.pt >= 25.,
                                                      op.abs(j.eta) <= 2.4)
        self.lambda_jetPUID = lambda j : op.OR(((j.puId >> 2) & 1) ,j.pt > 50.) # Jet PU ID bit1 is loose (only to be applied to jets with pt<50)

        self.ak4JetsPreSelForPUID = op.select(self.ak4JetsByPt, lambda j : op.AND(j.pt<=50.,self.lambda_ak4JetsPreSel(j)))
        self.ak4JetsPreSel        = op.select(self.ak4JetsByPt, lambda j : op.AND(self.lambda_ak4JetsPreSel(j),self.lambda_jetPUID(j)))



        # Cleaning #
        if self.args.POGID:
            self.lambda_cleanAk4Jets = lambda j : op.AND(op.NOT(op.rng_any(self.electronsTightSel, lambda ele : op.deltaR(j.p4, ele.p4) <= 0.4 )), 
                                                         op.NOT(op.rng_any(self.muonsTightSel, lambda mu : op.deltaR(j.p4, mu.p4) <= 0.4 )))
        elif self.args.TTHIDLoose or self.args.TTHIDTight:
            if channel == 'SL':
                def returnLambdaCleaningWithRespectToLeadingLepton(DR):
                   return lambda j : op.multiSwitch(
                            (op.AND(op.rng_len(self.electronsFakeSel) >= 1, op.rng_len(self.muonsFakeSel) == 0), op.deltaR(j.p4, self.electronsFakeSel[0].p4)>=DR),
                            (op.AND(op.rng_len(self.electronsFakeSel) == 0, op.rng_len(self.muonsFakeSel) >= 1), op.deltaR(j.p4, self.muonsFakeSel[0].p4)>=DR),
                            (op.AND(op.rng_len(self.muonsFakeSel) >= 1, op.rng_len(self.electronsFakeSel) >= 1),
                                   op.switch(self.electron_conept[self.electronsFakeSel[0].idx] >= self.muon_conept[self.muonsFakeSel[0].idx],
                                             op.deltaR(j.p4, self.electronsFakeSel[0].p4)>=DR,
                                             op.deltaR(j.p4, self.muonsFakeSel[0].p4)>=DR)),
                            op.c_bool(True))
                self.lambda_cleanAk4Jets = returnLambdaCleaningWithRespectToLeadingLepton(0.4)

            # remove jets within cone of DR<0.4 of leading lept
            if channel == 'DL':
                def returnLambdaCleaningWithRespectToLeadingLeptons(DR):
                    return lambda j : op.multiSwitch(
                          (op.AND(op.rng_len(self.electronsFakeSel) >= 2,op.rng_len(self.muonsFakeSel) == 0), 
                              # Only electrons 
                              op.AND(op.deltaR(j.p4, self.electronsFakeSel[0].p4)>=DR, op.deltaR(j.p4, self.electronsFakeSel[1].p4)>=DR)),
                          (op.AND(op.rng_len(self.electronsFakeSel) == 0,op.rng_len(self.muonsFakeSel) >= 2), 
                              # Only muons  
                              op.AND(op.deltaR(j.p4, self.muonsFakeSel[0].p4)>=DR, op.deltaR(j.p4, self.muonsFakeSel[1].p4)>=DR)),
                          (op.AND(op.rng_len(self.electronsFakeSel) == 1,op.rng_len(self.muonsFakeSel) == 1),
                              # One electron + one muon
                              op.AND(op.deltaR(j.p4, self.electronsFakeSel[0].p4)>=DR, op.deltaR(j.p4, self.muonsFakeSel[0].p4)>=DR)),
                          (op.AND(op.rng_len(self.electronsFakeSel) >= 1,op.rng_len(self.muonsFakeSel) >= 1),
                              # At least one electron + at least one muon
                           op.switch(self.electron_conept[self.electronsFakeSel[0].idx] > self.muon_conept[self.muonsFakeSel[0].idx],
                                     # Electron is leading #
                                     op.switch(op.rng_len(self.electronsFakeSel) == 1,
                                               op.AND(op.deltaR(j.p4, self.electronsFakeSel[0].p4)>=DR, op.deltaR(j.p4, self.muonsFakeSel[0].p4)>=DR),
                                               op.switch(self.electron_conept[self.electronsFakeSel[1].idx] > self.muon_conept[self.muonsFakeSel[0].idx],
                                                         op.AND(op.deltaR(j.p4, self.electronsFakeSel[0].p4)>=DR, op.deltaR(j.p4, self.electronsFakeSel[1].p4)>=DR),
                                                         op.AND(op.deltaR(j.p4, self.electronsFakeSel[0].p4)>=DR, op.deltaR(j.p4, self.muonsFakeSel[0].p4)>=DR))),
                                     # Muon is leading #
                                     op.switch(op.rng_len(self.muonsFakeSel) == 1,
                                               op.AND(op.deltaR(j.p4, self.muonsFakeSel[0].p4)>=DR, op.deltaR(j.p4, self.electronsFakeSel[0].p4)>=DR),
                                               op.switch(self.muon_conept[self.muonsFakeSel[1].idx] > self.electron_conept[self.electronsFakeSel[0].idx],
                                                         op.AND(op.deltaR(j.p4, self.muonsFakeSel[0].p4)>=DR, op.deltaR(j.p4, self.muonsFakeSel[1].p4)>=DR),
                                                         op.AND(op.deltaR(j.p4, self.muonsFakeSel[0].p4)>=DR, op.deltaR(j.p4, self.electronsFakeSel[0].p4)>=DR))))),
                           op.c_bool(True))
                self.lambda_cleanAk4Jets = returnLambdaCleaningWithRespectToLeadingLeptons(0.4)
            
        else:
            self.lambda_cleanAk4Jets = lambda j : op.c_float(True)

        self.ak4JetsForPUID     = op.select(self.ak4JetsPreSelForPUID,self.lambda_cleanAk4Jets) # Pt ordered
        self.ak4Jets            = op.select(self.ak4JetsPreSel,self.lambda_cleanAk4Jets) # Pt ordered
        self.ak4JetsByBtagScore = op.sort(self.ak4Jets, lambda j : -j.btagDeepFlavB) # Btag score ordered
    
        ############     Btagging     #############
        # The pfDeepFlavour (DeepJet) algorithm is used
        if era == "2016": 
            self.lambda_ak4BtagLoose =   lambda jet    : jet.btagDeepFlavB > 0.0614
            self.lambda_ak4Btag =   lambda jet    : jet.btagDeepFlavB > 0.3093
            self.lambda_ak4NoBtag = lambda jet    : jet.btagDeepFlavB <= 0.3093
        elif era =="2017":
            self.lambda_ak4BtagLoose =   lambda jet    : jet.btagDeepFlavB > 0.0521
            self.lambda_ak4Btag =   lambda jet    : jet.btagDeepFlavB > 0.3033
            self.lambda_ak4NoBtag = lambda jet    : jet.btagDeepFlavB <= 0.3033
        elif era == "2018":
            self.lambda_ak4BtagLoose =   lambda jet    : jet.btagDeepFlavB > 0.0494
            self.lambda_ak4Btag =   lambda jet    : jet.btagDeepFlavB > 0.2770
            self.lambda_ak4NoBtag = lambda jet    : jet.btagDeepFlavB <= 0.2770

        self.ak4BJets     = op.select(self.ak4Jets, self.lambda_ak4Btag)
        self.ak4BJetsLoose     = op.select(self.ak4Jets, self.lambda_ak4BtagLoose)
        self.ak4LightJetsByPt = op.select(self.ak4Jets, self.lambda_ak4NoBtag)
        self.ak4LightJetsByBtagScore = op.sort(self.ak4LightJetsByPt, lambda jet : -jet.btagDeepFlavB)
        # Sorted by btag score because for 0 and 1 btag categories, 

        # Doesn't contain the leading bTag scored Light Jet
        self.remainingJets = op.select(self.ak4LightJetsByPt, lambda jet : jet.idx != self.ak4LightJetsByBtagScore[0].idx)
        self.remainingJetPairs = lambda jets : op.combine(jets, N=2)
        # Wjj selection for resolved2b2j

        #############################################################################
        #                                AK8 Jets                                   #
        #############################################################################
        self.ak8JetsByPt = op.sort(t.FatJet, lambda jet : -jet.pt)
        self.ak8JetsByDeepB = op.sort(t.FatJet, lambda jet : -jet.btagDeepB)
        # Preselection #
        self.lambda_ak8JetsPreSel = lambda j : op.AND(j.jetId & 1 if era == "2016" else j.jetId & 2, # Jet ID flags bit1 is loose, bit2 is tight, bit3 is tightLepVeto
                                                      j.pt >= 200.,
                                                      op.abs(j.p4.Eta())<= 2.4,
                                                      op.AND(j.subJet1.isValid, j.subJet1.pt >= 20. , op.abs(j.subJet1.eta)<=2.4,
                                                             j.subJet2.isValid, j.subJet2.pt >= 20. , op.abs(j.subJet2.eta)<=2.4),
                                                             # Fatjet subjets must exist before checking Pt and eta 
                                                      op.AND(j.msoftdrop >= 30, j.msoftdrop <= 210),
                                                      j.tau2/j.tau1 <= 0.75)
        self.ak8JetsPreSel = op.select(self.ak8JetsByDeepB, self.lambda_ak8JetsPreSel) if channel == 'SL' else op.select(self.ak8JetsByPt, self.lambda_ak8JetsPreSel)

        # Cleaning #
        if self.args.POGID:
            self.lambda_cleanAk8Jets = lambda j : op.AND(op.NOT(op.rng_any(self.electronsTightSel, lambda ele : op.deltaR(j.p4, ele.p4) <= 0.8 )), 
                                                         op.NOT(op.rng_any(self.muonsTightSel, lambda mu : op.deltaR(j.p4, mu.p4) <= 0.8 )))
        elif self.args.TTHIDLoose or self.args.TTHIDTight:
            if channel == 'SL':
                self.lambda_cleanAk8Jets = returnLambdaCleaningWithRespectToLeadingLepton(0.8)
            if channel == 'DL':
                self.lambda_cleanAk8Jets = returnLambdaCleaningWithRespectToLeadingLeptons(0.8)
        # remove jets within cone of DR<0.8 of preselected electrons and muons
        else:
            self.lambda_cleanAk8Jets = lambda j : op.c_float(True)
        self.ak8Jets = op.select(self.ak8JetsPreSel,self.lambda_cleanAk8Jets)

        ############     Btagging     #############
        # The DeepCSV b-tagging algorithm is used on subjets 
        if era == "2016": 
            self.lambda_subjetBtag = lambda subjet: subjet.btagDeepB > 0.6321
        elif era =="2017":
            self.lambda_subjetBtag = lambda subjet: subjet.btagDeepB > 0.4941
        elif era == "2018":
            self.lambda_subjetBtag = lambda subjet: subjet.btagDeepB > 0.4184

        self.lambda_ak8Btag = lambda fatjet : op.OR(op.AND(fatjet.subJet1.pt >= 30, self.lambda_subjetBtag(fatjet.subJet1)),
                                                    op.AND(fatjet.subJet2.pt >= 30, self.lambda_subjetBtag(fatjet.subJet2)))
        self.lambda_ak8noBtag = lambda fatjet : op.NOT(op.OR(op.AND(fatjet.subJet1.pt >= 30, self.lambda_subjetBtag(fatjet.subJet1)),
                                                       op.AND(fatjet.subJet2.pt >= 30, self.lambda_subjetBtag(fatjet.subJet2))))
        self.lambda_ak8Btag_bothSubJets = lambda fatjet : op.AND(op.AND(fatjet.subJet1.pt >= 30, self.lambda_subjetBtag(fatjet.subJet1)),
                                                                 op.AND(fatjet.subJet2.pt >= 30, self.lambda_subjetBtag(fatjet.subJet2)))

        self.ak8BJets = op.select(self.ak8Jets, self.lambda_ak8Btag)
        self.ak8nonBJets = op.select(self.ak8Jets, self.lambda_ak8noBtag)
        # Ak4 Jet Collection cleaned from Ak8b #
        self.lambda_cleanAk4FromAk8b = lambda ak4j : op.NOT(op.AND(op.rng_len(self.ak8BJets) > 0, op.deltaR(ak4j.p4,self.ak8BJets[0].p4) <= 0.8))
        self.ak4JetsCleanedFromAk8b  = op.select(self.ak4LightJetsByPt, self.lambda_cleanAk4FromAk8b)

        # used as a BDT input for SemiBoosted category
        self.lambda_btaggedSubJets = lambda fjet : op.switch(self.lambda_ak8Btag_bothSubJets(fjet), op.c_float(2.0), op.c_float(1.0))
        self.nMediumBTaggedSubJets = op.rng_sum(self.ak8BJets, self.lambda_btaggedSubJets)

        #############################################################################
        #                                VBF Jets                                   #
        #############################################################################
        self.lambda_VBFJets = lambda j : op.AND(j.jetId & 1 if era == "2016" else j.jetId & 2, # Jet ID flags bit1 is loose, bit2 is tight, bit3 is tightLepVeto
                                                j.pt >= 30.,
                                                op.abs(j.eta) <= 4.7,
                                                op.OR(j.pt >= 60.,
                                                      op.abs(j.eta) < 2.7, 
                                                      op.abs(j.eta) > 3.0))
        self.VBFJetsPreSelForPUID = op.select(self.ak4JetsByPt, lambda j : op.AND(j.pt<=50.,self.lambda_VBFJets(j)))
        self.VBFJetsPreSel        = op.select(self.ak4JetsByPt, lambda j : op.AND(self.lambda_VBFJets(j),self.lambda_jetPUID(j)))

        if self.args.POGID:
            self.lambda_cleanVBFLeptons = lambda j : op.AND(op.NOT(op.rng_any(self.electronsTightSel, lambda ele : op.deltaR(j.p4, ele.p4) <= 0.4 )), 
                                                            op.NOT(op.rng_any(self.muonsTightSel, lambda mu : op.deltaR(j.p4, mu.p4) <= 0.4 )))
        elif self.args.TTHIDLoose or self.args.TTHIDTight:
            if channel == 'SL':
                self.lambda_cleanVBFLeptons = returnLambdaCleaningWithRespectToLeadingLepton(0.4)
            if channel == 'DL':
                self.lambda_cleanVBFLeptons = returnLambdaCleaningWithRespectToLeadingLeptons(0.4)
        else:
            self.lambda_cleanVBFLeptons = lambda j : op.c_bool(True)

        self.VBFJetsForPUID     = op.select(self.VBFJetsPreSelForPUID, self.lambda_cleanVBFLeptons)
        self.VBFJets            = op.select(self.VBFJetsPreSel, self.lambda_cleanVBFLeptons)

        self.lambda_VBFPair     = lambda j1,j2 : op.AND(op.invariant_mass(j1.p4,j2.p4) > 500.,
                                                        op.abs(j1.eta - j2.eta) > 3.)
        
        if channel == "DL":
            self.lambda_cleanVBFAk4 = lambda j : op.multiSwitch((op.rng_len(self.ak4JetsByBtagScore)>1,op.AND(op.deltaR(j.p4, self.ak4JetsByBtagScore[0].p4)>0.8,
                                                                                                              op.deltaR(j.p4, self.ak4JetsByBtagScore[1].p4)>0.8)),
                                                                (op.rng_len(self.ak4JetsByBtagScore)==1,op.deltaR(j.p4, self.ak4JetsByBtagScore[0].p4)>0.8),
                                                                op.c_bool(True))
            #op.AND(op.NOT(op.rng_any(self.ak4JetsByBtagScore[:2], lambda ak4Jet : op.deltaR(j.p4, ak4Jet.p4) <= 0.8 )))
            self.lambda_cleanVBFAk8 = lambda j : op.multiSwitch((op.rng_len(self.ak8BJets)>0,op.deltaR(j.p4, self.ak8BJets[0].p4) > 1.2),
                                                                (op.rng_len(self.ak8Jets)>0,op.deltaR(j.p4, self.ak8Jets[0].p4) > 1.2),
                                                                op.c_bool(True))


            self.VBFJetsResolved = op.select(self.VBFJets, self.lambda_cleanVBFAk4)
            self.VBFJetsBoosted  = op.select(self.VBFJets, self.lambda_cleanVBFAk8)

            self.VBFJetPairs         = op.sort(op.combine(self.VBFJets, N=2, pred=self.lambda_VBFPair), lambda dijet : -op.invariant_mass(dijet[0].p4,dijet[1].p4))
            self.VBFJetPairsResolved = op.sort(op.combine(self.VBFJetsResolved, N=2, pred=self.lambda_VBFPair), lambda dijet : -op.invariant_mass(dijet[0].p4,dijet[1].p4))
            self.VBFJetPairsBoosted  = op.sort(op.combine(self.VBFJetsBoosted,  N=2, pred=self.lambda_VBFPair), lambda dijet : -op.invariant_mass(dijet[0].p4,dijet[1].p4))

        if channel == "SL":
            #def cleanVBFwithJPA_Resolved(self, jpaJets, nJpaJets):
            #    return lambda j : op.OR(*(op.deltaR(jpaJets[i].p4, j.p4) > 0.8 for i in range(nJpaJets)))
            #def cleanVBFwithJPA_Boosted(self, jpaJets, nJpaJets):
            #    return lambda j : op.AND(op.rng_len(self.ak8BJets) >= 1, op.OR(op.OR(*(op.deltaR(jpaJets[i].p4, j.p4) > 0.8 for i in range(nJpaJets))),
            #                                                                   op.deltaR(self.ak8Jets[0].p4, j.p4) > 1.2))
            print('VBF <<>> ak4/8-jets  cleaning is in Skimmer now!!!')

        #############################################################################
        #                             Scalefactors                                  #
        #############################################################################
        self.SF = ScaleFactorsbbWW()    
        if self.is_MC:
            #---- PU ID SF ----#
            # Efficiency and mistags do not have uncertainties, the systematics are in the SF 
            self.jetpuid_mc_eff = self.SF.get_scalefactor("lepton", ('jet_puid_eff','eff_{}_L'.format(era)),combine="weight", defineOnFirstUse=(not forSkimmer))
            self.jetpuid_mc_mis = self.SF.get_scalefactor("lepton", ('jet_puid_eff','mistag_{}_L'.format(era)),combine="weight", defineOnFirstUse=(not forSkimmer))
            self.jetpuid_sf_eff = self.SF.get_scalefactor("lepton", ('jet_puid_sf','eff_{}_L'.format(era)),combine="weight", systName="jetpuid_eff_sf", defineOnFirstUse=(not forSkimmer))
            self.jetpuid_sf_mis = self.SF.get_scalefactor("lepton", ('jet_puid_sf','mistag_{}_L'.format(era)),combine="weight", systName="jetpuid_mistag_sf", defineOnFirstUse=(not forSkimmer))

            #---- Object SF -----# (Will take as argument the era)
            ####  Muons ####
            self.muLooseId = self.SF.get_scalefactor("lepton", 'muon_loose_{}'.format(era), combine="weight", systName="mu_loose", defineOnFirstUse=(not forSkimmer)) 
            self.lambda_MuonLooseSF = lambda mu : [op.switch(self.lambda_is_matched(mu), self.muLooseId(mu), op.c_float(1.))]
            if self.args.TTHIDTight:
                self.muTightMVA = self.SF.get_scalefactor("lepton", 'muon_tightMVA_{}'.format(era), combine="weight", systName="mu_tightmva", defineOnFirstUse=(not forSkimmer))
            if self.args.TTHIDLoose:
                self.muTightMVA = self.SF.get_scalefactor("lepton", 'muon_tightMVArelaxed_{}'.format(era), combine="weight", systName="mu_tightmva", defineOnFirstUse=(not forSkimmer))

            self.lambda_MuonTightSF = lambda mu : [op.switch(self.lambda_muonTightSel(mu),      # check if actual tight lepton (needed because in Fake CR one of the dilepton is not)
                                                             op.switch(self.lambda_is_matched(mu), # check if gen matched
                                                                       self.muTightMVA(mu),
                                                                       op.c_float(1.)),
                                                             op.c_float(1.))]

            self.muPOGTightID = self.SF.get_scalefactor("lepton", 'muon_POGSF_ID_{}'.format(era), combine="weight", systName="mu_pogtightid", defineOnFirstUse=(not forSkimmer))
            self.muPOGTightISO = self.SF.get_scalefactor("lepton", 'muon_POGSF_ISO_{}'.format(era), combine="weight", systName="mu_pogtightiso", defineOnFirstUse=(not forSkimmer))
            self.elPOGTight = self.SF.get_scalefactor("lepton", 'electron_POGSF_{}'.format(era), combine="weight", systName="el_pogtight", defineOnFirstUse=(not forSkimmer))

            ####  Electrons ####
            if era == "2016" or era == "2017": # Electron reco eff depend on Pt for 2016 and 2017
                self.elLooseRecoPtLt20 = self.SF.get_scalefactor("lepton", ('electron_loosereco_{}'.format(era) , 'electron_loosereco_ptlt20'), combine="weight", systName="el_looserecoptlt20", defineOnFirstUse=(not forSkimmer))
                self.elLooseRecoPtGt20 = self.SF.get_scalefactor("lepton", ('electron_loosereco_{}'.format(era) , 'electron_loosereco_ptgt20'), combine="weight", systName="el_looserecoptgt20", defineOnFirstUse=(not forSkimmer))
                # /!\ In analysis YAML file 2016 and 2017 for systematics : must use el_looserecoptlt20 and el_looserecoptgt20

            elif era == "2018": # Does not depend on pt for 2018
                self.elLooseReco = self.SF.get_scalefactor("lepton", 'electron_loosereco_{}'.format(era), combine="weight", systName="el_loosereco", defineOnFirstUse=(not forSkimmer))
                # /!\ In analysis YAML file 2018 for systematics : must use el_loosereco

            self.elLooseId = self.SF.get_scalefactor("lepton", 'electron_looseid_{}'.format(era) , combine="weight", systName="el_looseid", defineOnFirstUse=(not forSkimmer))
            self.elLooseEff = self.SF.get_scalefactor("lepton", 'electron_looseeff_{}'.format(era) , combine="weight", systName="el_looseeff", defineOnFirstUse=(not forSkimmer))

            if era == "2016" or era == "2017":
                self.lambda_ElectronLooseSF = lambda el : [op.switch(self.lambda_is_matched(el),self.elLooseId(el),op.c_float(1.)), 
                                                           op.switch(self.lambda_is_matched(el),self.elLooseEff(el),op.c_float(1.)), 
                                                           op.switch(self.lambda_is_matched(el),op.switch(el.pt>20, self.elLooseRecoPtGt20(el), self.elLooseRecoPtLt20(el)),op.c_float(1.))]
            elif era == "2018": 
                self.lambda_ElectronLooseSF = lambda el : [op.switch(self.lambda_is_matched(el),self.elLooseId(el),op.c_float(1.)),
                                                           op.switch(self.lambda_is_matched(el),self.elLooseEff(el),op.c_float(1.)),
                                                           op.switch(self.lambda_is_matched(el),self.elLooseReco(el),op.c_float(1.))]

            if self.args.TTHIDTight:
                self.elTightMVA = self.SF.get_scalefactor("lepton", 'electron_tightMVA_{}'.format(era) , combine="weight", systName="el_tightmva", defineOnFirstUse=(not forSkimmer))
            if self.args.TTHIDLoose:
                self.elTightMVA = self.SF.get_scalefactor("lepton", 'electron_tightMVArelaxed_{}'.format(era) , combine="weight", systName="el_tightmva", defineOnFirstUse=(not forSkimmer))

            self.lambda_ElectronTightSF = lambda el : [op.switch(self.lambda_electronTightSel(el),      # check if actual tight lepton (needed because in Fake CR one of the dilepton is not)
                                                                 op.switch(self.lambda_is_matched(el), # check if gen matched
                                                                           self.elTightMVA(el),
                                                                           op.c_float(1.)),
                                                                 op.c_float(1.))]

#            #### Ak8 btag efficiencies ####
#            if self.isNanov7:
#                self.Ak8Eff_bjets = self.SF.get_scalefactor("lepton",('ak8btag_eff_{}'.format(era),'eff_bjets'),combine="weight", systName="ak8btag_eff_bjets", defineOnFirstUse=(not forSkimmer),
#                                                            additionalVariables={'Pt':lambda x : x.pt,'Eta':lambda x : x.eta, 'BTagDiscri':lambda x : x.btagDeepB})
#                self.Ak8Eff_cjets = self.SF.get_scalefactor("lepton",('ak8btag_eff_{}'.format(era),'eff_cjets'),combine="weight", systName="ak8btag_eff_cjets", defineOnFirstUse=(not forSkimmer),
#                                                            additionalVariables={'Pt':lambda x : x.pt,'Eta':lambda x : x.eta, 'BTagDiscri':lambda x : x.btagDeepB})
#                self.Ak8Eff_lightjets = self.SF.get_scalefactor("lepton",('ak8btag_eff_{}'.format(era),'eff_lightjets'),combine="weight", systName="ak8btag_eff_lightjets", defineOnFirstUse=(not forSkimmer),
#                                                            additionalVariables={'Pt':lambda x : x.pt,'Eta':lambda x : x.eta, 'BTagDiscri':lambda x : x.btagDeepB})
 
            #----- Triggers -----# 
            #### Single lepton triggers ####
            self.ttH_singleElectron_trigSF = self.SF.get_scalefactor("lepton", 'singleTrigger_electron_{}'.format(era) , combine="weight", systName="ttH_singleElectron_trigSF", defineOnFirstUse=(not forSkimmer))
            self.ttH_singleMuon_trigSF = self.SF.get_scalefactor("lepton", 'singleTrigger_muon_{}'.format(era) , combine="weight", systName="ttH_singleMuon_trigSF", defineOnFirstUse=(not forSkimmer))
            
            #### Double lepton triggers #### (Need to split according to era) 
                # https://gitlab.cern.ch/ttH_leptons/doc/-/blob/master/Legacy/data_to_mc_corrections.md#trigger-efficiency-scale-factors -> deprecated
                # New ref (more up to date) : https://cernbox.cern.ch/index.php/s/lW2BiTli5tJR0MN 
                # Same but clearer  : http://cms.cern.ch/iCMS/jsp/iCMS.jsp?mode=single&part=publications (AN-2019/111) at Table 30, page 43
                # -> Based on subleading conept lepton (except mumu 2018), use lambda 
                # Lambdas return list to be easily concatenated with other SF in lists as well (the weight argument of op.refine requires a list)
            if era == "2016":
                # Double Electron #
                self.lambda_ttH_doubleElectron_trigSF = lambda dilep : op.multiSwitch(
                    (self.electron_conept[dilep[1].idx]<25 , op.systematic(op.c_float(0.98), name="ttH_doubleElectron_trigSF", up=op.c_float(0.98*1.02), down=op.c_float(0.98*0.98))), 
                    op.systematic(op.c_float(1.), name="ttH_doubleElectron_trigSF", up=op.c_float(1.02), down=op.c_float(0.98)))
                # Double Muon #
                self.lambda_ttH_doubleMuon_trigSF = lambda dilep : op.systematic(op.c_float(0.99), name="ttH_doubleMuon_trigSF", up=op.c_float(0.99*1.01), down=op.c_float(0.99*0.99))
                # Electron Muon #
                self.lambda_ttH_electronMuon_trigSF = lambda dilep : op.systematic(op.c_float(1.00), name="ttH_electronMuon_trigSF", up=op.c_float(1.01), down=op.c_float(0.99))
            elif era == "2017":
                # Double Electron #
                self.lambda_ttH_doubleElectron_trigSF = lambda dilep : op.multiSwitch(
                    (self.electron_conept[dilep[1].idx]<40 , op.systematic(op.c_float(0.98), name="ttH_doubleElectron_trigSF", up=op.c_float(0.98*1.01), down=op.c_float(0.98*0.99))), 
                    op.systematic(op.c_float(1.), name="ttH_doubleElectron_trigSF", up=op.c_float(1.01), down=op.c_float(0.99)))
                # Double Muon #
                self.lambda_ttH_doubleMuon_trigSF = lambda dilep : op.multiSwitch(
                    (self.muon_conept[dilep[1].idx]<40 , op.systematic(op.c_float(0.97), name="ttH_doubleMuon_trigSF", up=op.c_float(0.97*1.02), down=op.c_float(0.97*0.98))),
                    (self.muon_conept[dilep[1].idx]<55 , op.systematic(op.c_float(0.995), name="ttH_doubleMuon_trigSF", up=op.c_float(0.995*1.02), down=op.c_float(0.995*0.98))),
                    (self.muon_conept[dilep[1].idx]<70 , op.systematic(op.c_float(0.96), name="ttH_doubleMuon_trigSF", up=op.c_float(0.96*1.02), down=op.c_float(0.96*0.98))),
                    op.systematic(op.c_float(0.94), name="ttH_doubleMuon_trigSF", up=op.c_float(0.94*1.02), down=op.c_float(0.94*0.98)))
                # Electron Muon : /!\ While ElEl and MuMu is conept ordered, ElMu is not #
                self.lambda_ttH_electronMuon_trigSF = lambda dilep : op.multiSwitch(
                    (op.min(self.electron_conept[dilep[0].idx],self.muon_conept[dilep[1].idx])<40, op.systematic(op.c_float(0.98), name="ttH_electronMuon_trigSF", up=op.c_float(0.98*1.01), down=op.c_float(0.98*0.99))),
                    op.systematic(op.c_float(0.99), name="ttH_electronMuon_trigSF", up=op.c_float(0.99*1.01), down=op.c_float(0.99*0.99)))
            elif era == "2018":
                # Double Electron #
                self.lambda_ttH_doubleElectron_trigSF = lambda dilep : op.multiSwitch(
                    (self.electron_conept[dilep[1].idx]<25 , op.systematic(op.c_float(0.98), name="ttH_doubleElectron_trigSF", up=op.c_float(0.98*1.01), down=op.c_float(0.98*0.99))), 
                    op.systematic(op.c_float(1.), name="ttH_doubleElectron_trigSF", up=op.c_float(1.01), down=op.c_float(0.99)))
                # Double Muon /!\ only this one using leading conept lepton #
                self.lambda_ttH_doubleMuon_trigSF = lambda dilep : op.multiSwitch(
                    (self.muon_conept[dilep[0].idx]<40 , op.systematic(op.c_float(1.01), name="ttH_doubleMuon_trigSF", up=op.c_float(1.01*1.01), down=op.c_float(1.01*0.99))),
                    (self.muon_conept[dilep[0].idx]<70 , op.systematic(op.c_float(0.995), name="ttH_doubleMuon_trigSF", up=op.c_float(0.995*1.01), down=op.c_float(0.995*0.99))),
                    op.systematic(op.c_float(0.98), name="ttH_doubleMuon_trigSF", up=op.c_float(0.98*1.01), down=op.c_float(0.98*0.99)))
                # Electron Muon : /!\ While ElEl and MuMu is conept ordered, ElMu is not #
                self.lambda_ttH_electronMuon_trigSF = lambda dilep : op.multiSwitch(
                    (op.min(self.electron_conept[dilep[0].idx],self.muon_conept[dilep[1].idx])<25, op.systematic(op.c_float(0.98), name="ttH_electronMuon_trigSF", up=op.c_float(0.98*1.01), down=op.c_float(0.98*0.99))),
                    op.systematic(op.c_float(1.00), name="ttH_electronMuon_trigSF", up=op.c_float(1.01), down=op.c_float(0.99)))
        #----- Fake rates -----#
        if self.args.TTHIDTight or self.args.TTHIDLoose:
            if self.args.TTHIDTight:
                FRSysts = ['Tight_pt_syst','Tight_barrel_syst','Tight_norm_syst']
            if self.args.TTHIDLoose:
                FRSysts = ['Loose_pt_syst','Loose_barrel_syst','Loose_norm_syst']
            self.electronFRList = [self.SF.get_scalefactor("lepton", ('electron_fakerates_'+era, syst), combine="weight", systName="el_FR_"+syst, defineOnFirstUse=(not forSkimmer),
                                                 additionalVariables={'Pt' : lambda obj : self.electron_conept[obj.idx]}) for syst in FRSysts]
            self.muonFRList = [self.SF.get_scalefactor("lepton", ('muon_fakerates_'+era, syst), combine="weight", systName="mu_FR_"+syst, defineOnFirstUse=(not forSkimmer),
                                             additionalVariables={'Pt' : lambda obj : self.muon_conept[obj.idx]}) for syst in FRSysts ] 

            def returnFFSF(obj,list_SF,systName):
                """ Helper when several systematics are present  """
                args = [ a(obj) for a in list_SF[0]._args ] ## get the things the SF depends on
                systArgs = {'nominal':list_SF[0].sfOp.get(*(args+[op.extVar("int", "Nominal")])),'name':systName}
                systArgs.update({SF._systName+'up':SF.sfOp.get(*(args+[op.extVar("int", "Up")])) for SF in list_SF})
                systArgs.update({SF._systName+'down':SF.sfOp.get(*(args+[op.extVar("int", "Down")])) for SF in list_SF})
                return op.systematic(**systArgs)


        #############################################################################
        #                             High level lambdas                            #
        #############################################################################
        self.HLL = highlevelLambdas(self)
        
        #############################################################################
        #                             Fake Factors                                  #
        #############################################################################
        if self.args.TTHIDTight or self.args.TTHIDLoose:
            # Non closure between ttbar and QCD correction #
            # https://gitlab.cern.ch/cms-hh-bbww/cms-hh-to-bbww/-/blob/master/Legacy/backgroundEstimation.md
            if era == '2016':
                self.electronCorrFR = op.systematic(op.c_float(1.376), name="electronCorrFR",up=op.c_float(1.376*1.376),down=op.c_float(1.))
                self.muonCorrFR     = op.systematic(op.c_float(1.050), name="muonCorrFR",up=op.c_float(1.050*1.050),down=op.c_float(1.))
            elif era == '2017':
                self.electronCorrFR = op.systematic(op.c_float(1.252), name="electronCorrFR",up=op.c_float(1.252*1.252),down=op.c_float(1.))
                self.muonCorrFR     = op.systematic(op.c_float(1.157), name="muonCorrFR",up=op.c_float(1.157*1.157),down=op.c_float(1.))
            elif era == '2018':
                self.electronCorrFR = op.systematic(op.c_float(1.325), name="electronCorrFR",up=op.c_float(1.325*1.325),down=op.c_float(1.))
                self.muonCorrFR     = op.systematic(op.c_float(1.067), name="muonCorrFR",up=op.c_float(1.067*1.067),down=op.c_float(1.))

            self.lambda_FR_el       = lambda el : returnFFSF(el,self.electronFRList,"el_FR")
            self.lambda_FR_mu       = lambda mu : returnFFSF(mu,self.muonFRList,"mu_FR")
            self.lambda_FRcorr_el   = lambda el : self.lambda_FR_el(el)*self.electronCorrFR
            self.lambda_FRcorr_mu   = lambda mu : self.lambda_FR_mu(mu)*self.muonCorrFR
            self.lambda_FF_el       = lambda el : self.lambda_FRcorr_el(el)/(1-self.lambda_FRcorr_el(el))
            self.lambda_FF_mu       = lambda mu : self.lambda_FRcorr_mu(mu)/(1-self.lambda_FRcorr_mu(mu))

            if channel == "SL":
                self.ElFakeFactor = lambda el : self.lambda_FF_el(el)
                self.MuFakeFactor = lambda mu : self.lambda_FF_mu(mu)
            if channel == "DL":
                self.ElElFakeFactor = lambda dilep : op.multiSwitch((op.AND(op.NOT(self.lambda_electronTightSel(dilep[0])),op.NOT(self.lambda_electronTightSel(dilep[1]))),
                                                                     # Both electrons fail tight -> -F1*F2
                                                                     -self.lambda_FF_el(dilep[0])*self.lambda_FF_el(dilep[1])),
                                                                    (op.AND(op.NOT(self.lambda_electronTightSel(dilep[0])),self.lambda_electronTightSel(dilep[1])),
                                                                     # Only leading electron fails tight -> F1
                                                                     self.lambda_FF_el(dilep[0])),
                                                                    (op.AND(self.lambda_electronTightSel(dilep[0]),op.NOT(self.lambda_electronTightSel(dilep[1]))),
                                                                     # Only subleading electron fails tight -> F2
                                                                     self.lambda_FF_el(dilep[1])),
                                                                     op.c_float(1.)) # Should not happen
                self.MuMuFakeFactor = lambda dilep : op.multiSwitch((op.AND(op.NOT(self.lambda_muonTightSel(dilep[0])),op.NOT(self.lambda_muonTightSel(dilep[1]))),
                                                                     # Both muons fail tight -> -F1*F2
                                                                     -self.lambda_FF_mu(dilep[0])*self.lambda_FF_mu(dilep[1])),
                                                                    (op.AND(op.NOT(self.lambda_muonTightSel(dilep[0])),self.lambda_muonTightSel(dilep[1])),
                                                                     # Only leading muon fails tight -> F1
                                                                     self.lambda_FF_mu(dilep[0])),
                                                                    (op.AND(self.lambda_muonTightSel(dilep[0]),op.NOT(self.lambda_muonTightSel(dilep[1]))),
                                                                     # Only subleading muon fails tight -> F2
                                                                     self.lambda_FF_mu(dilep[1])),
                                                                     op.c_float(1.)) # Should not happen
                self.ElMuFakeFactor = lambda dilep : op.multiSwitch((op.AND(op.NOT(self.lambda_electronTightSel(dilep[0])),op.NOT(self.lambda_muonTightSel(dilep[1]))),
                                                                     # Both electron and muon fail tight -> -F1*F2
                                                                     -self.lambda_FF_el(dilep[0])*self.lambda_FF_mu(dilep[1])),
                                                                    (op.AND(op.NOT(self.lambda_electronTightSel(dilep[0])),self.lambda_muonTightSel(dilep[1])),
                                                                     # Only electron fails tight -> F1
                                                                     self.lambda_FF_el(dilep[0])),
                                                                    (op.AND(self.lambda_electronTightSel(dilep[0]),op.NOT(self.lambda_muonTightSel(dilep[1]))),
                                                                     # Only subleading electron fails tight -> F2
                                                                     self.lambda_FF_mu(dilep[1])),
                                                                     op.c_float(1.)) # Should not happen

        ###########################################################################
        #                    b-tagging efficiency scale factors                   #
        ###########################################################################
        if self.is_MC:
            if self.era == '2016':
                if ('db' in sampleCfg.keys() and 'TuneCP5' in sampleCfg['db']) or ('files' in sampleCfg.keys() and all(['TuneCP5' in f for f in sampleCfg['files']])):
                    csvFileNameAk4 = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "ScaleFactors_POG" , "DeepJet_2016LegacySF_V1_TuneCP5.csv")
                else: # With CUETP8M1
                    csvFileNameAk4 = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "ScaleFactors_POG" , "DeepJet_2016LegacySF_V1.csv")
            if self.era == '2017':
                csvFileNameAk4 = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "ScaleFactors_POG" , "DeepFlavour_94XSF_V4_B_F.csv")
            if self.era == '2018':
                csvFileNameAk4 = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "ScaleFactors_POG" , "DeepJet_102XSF_V2.csv")
                
            if not os.path.exists(csvFileNameAk4):
                raise RuntimeError('Could not find Ak4 csv file %s'%csvFileNameAk4)
            print ('Btag Ak4 CSV file',csvFileNameAk4)

            #----- Ak4 SF -----# 
            systTypes = ["jes", "lf", "hf", "hfstats1", "hfstats2","lfstats1", "lfstats2", "cferr1", "cferr2"]
                # From https://twiki.cern.ch/twiki/bin/view/CMS/BTagShapeCalibration#Systematic_uncertainties
            self.DeepJetDiscReshapingSF = BtagSF(taggerName       = "deepjet", 
                                                 csvFileName      = csvFileNameAk4,
                                                 wp               = "reshaping",  # "loose", "medium", "tight" or "reshaping"
                                                 sysType          = "central", 
                                                 otherSysTypes    = ['up_'+s for s in systTypes]+['down_'+s for s in systTypes], 
                                                 measurementType  = "iterativefit", 
                                                 sel              = noSel, 
                                                 getters          = {'Discri':lambda j : j.btagDeepFlavB},
                                                 uName            = self.sample)

#        if self.isNanov7:
#            if era == '2016':
#                csvFileNameAk8= os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "ScaleFactors_POG" , "subjet_DeepCSV_2016LegacySF_V1.csv")
#            if era == '2017':
#                csvFileNameAk8 = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "ScaleFactors_POG" , "subjet_DeepCSV_94XSF_V4_B_F_v2.csv")
#            if era == '2018':
#                csvFileNameAk8 = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", "ScaleFactors_POG" , "subjet_DeepCSV_102XSF_V1.csv")
#            
#            #----- Ak8 SF -----# 
#            if not os.path.exists(csvFileNameAk8):
#                raise RuntimeError('Could not find Ak8 csv file %s'%csvFileNameAk8)
#            print ('Btag Ak8 CSV file',csvFileNameAk8)
#            self.DeepCsvSubjetMediumSF = BtagSF(taggerName       = "deepcsvSubjet", 
#                                                csvFileName      = csvFileNameAk8,
#                                                wp               = "medium",
#                                                sysType          = "central", 
#                                                otherSysTypes    = ['up','down'],
#                                                measurementType  = {"B": "lt", "C": "lt", "UDSG": "incl"},
#                                                getters          = {'Discri':lambda subjet : subjet.btagDeepB,
#                                                                    'JetFlavour': lambda subjet : op.static_cast("BTagEntry::JetFlavor",
#                                                                                                        op.multiSwitch((subjet.nBHadrons>0,op.c_int(0)), # B -> flav = 5 -> BTV = 0
#                                                                                                                       (subjet.nCHadrons>0,op.c_int(1)), # C -> flav = 4 -> BTV = 1
#                                                                                                                       op.c_int(2)))},                  # UDSG -> flav = 0 -> BTV = 2
#                                                sel              = sel, 
#                                                uName            = self.sample)


        ###########################################################################
        #                                 RETURN                                  # 
        ###########################################################################
        return noSel



    def beforeJetselection(self,sel,name=''):
        ##############################################################################
        #                             Jets forceDefines                              #
        ##############################################################################
        from bamboo.analysisutils import forceDefine
        # Forcedefine : calculate once per event (for every event) #
        if not self.args.Synchronization:
            #forceDefine(t._Muon.calcProd, sel) # Muons for Rochester corrections
            forceDefine(self.tree._Jet.calcProd, sel)  # Jets for configureJets
            forceDefine(self.tree._FatJet.calcProd, sel)  # FatJets for configureJets
            forceDefine(getattr(self.tree, "_{0}".format("MET" if self.era != "2017" else "METFixEE2017")).calcProd,sel) # MET for configureMET
        else:
            print ("No jet corrections applied")

        #############################################################################
        #                           Jet PU ID reweighting                           #
        #############################################################################
        if self.is_MC:
            wFail = op.extMethod("scalefactorWeightForFailingObject", returnType="double")
            lambda_puid_weight = lambda j : op.switch(j.genJet.isValid,
                                                      op.switch(((j.puId >> 2) & 1),
                                                                self.jetpuid_sf_eff(j), 
                                                                wFail(self.jetpuid_sf_eff(j), self.jetpuid_mc_eff(j))),
                                                      op.switch(((j.puId >> 2) & 1),
                                                                self.jetpuid_sf_mis(j), 
                                                                wFail(self.jetpuid_sf_mis(j), self.jetpuid_mc_mis(j))))
            lambda_puid_efficiency = lambda j : op.switch(j.genJet.isValid,
                                                          op.switch(((j.puId >> 2) & 1),
                                                                    self.jetpuid_sf_eff(j),
                                                                    wFail(self.jetpuid_sf_eff(j), self.jetpuid_mc_eff(j))),
                                                          op.c_float(1.))
            lambda_puid_mistag     = lambda j : op.switch(j.genJet.isValid,
                                                          op.c_float(1.),
                                                          op.switch(((j.puId >> 2) & 1),
                                                                    self.jetpuid_sf_mis(j), 
                                                                    wFail(self.jetpuid_sf_mis(j), self.jetpuid_mc_mis(j))))

            self.puid_reweighting = op.rng_product(self.ak4JetsForPUID, lambda j : lambda_puid_weight(j)) * op.rng_product(self.VBFJetsForPUID, lambda j : lambda_puid_weight(j))
            self.puid_reweighting_efficiency = op.rng_product(self.ak4JetsForPUID, lambda j : lambda_puid_efficiency(j)) * op.rng_product(self.VBFJetsForPUID, lambda j : lambda_puid_efficiency(j))
                    # Sync purposes
            self.puid_reweighting_mistag = op.rng_product(self.ak4JetsForPUID, lambda j : lambda_puid_mistag(j)) * op.rng_product(self.VBFJetsForPUID, lambda j : lambda_puid_mistag(j))
                    # Sync purposes
       
            sel = sel.refine("jetPUIDReweighting"+name,weight=self.puid_reweighting)

        ###########################################################################
        #                    b-tagging efficiency scale factors                   #
        ###########################################################################
        if self.is_MC:
            #----- AK4 jets -> using Method 1.d -----#
            # See https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagShapeCalibration
            # W_btag = Π_i(all jets) SD(jet_i)  which must be multiplied by r = Σ w(before)/Σ w(after) (before/after using the btag weight, no btag selection for both)
                
            self.btagAk4SF = op.rng_product(self.ak4Jets , lambda j : self.DeepJetDiscReshapingSF(j))

            if self.args.BtagReweightingOn and self.args.BtagReweightingOff: 
                raise RuntimeError("Reweighting cannot be both on and off") 
            if self.args.BtagReweightingOn:
                sel = sel.refine("BtagSF" , weight = self.btagAk4SF)
            elif self.args.BtagReweightingOff:
                pass # Do not apply any SF
            else:
                ReweightingFileName = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','ScaleFactors_Btag','BtagReweightingRatio_jetN_{}_{}.json'.format(self.sample,self.era))
                if not os.path.exists(ReweightingFileName):
                    raise RuntimeError("Could not find reweighting file %s"%ReweightingFileName)
                print ('Reweighting file',ReweightingFileName)

                self.BtagRatioWeight = makeBtagRatioReweighting(jsonFile = ReweightingFileName,
                                                                numJets  = op.rng_len(self.ak4Jets),
                                                                nameHint = "bamboo_nJetsWeight_{}".format(self.sample.replace('-','_')))
                sel = sel.refine("BtagAk4SF"+name , weight = [self.btagAk4SF,self.BtagRatioWeight])

#            if self.isNanov7:
#                #----- AK8 jets -> using Method 1.a -----#
#                # See https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagShapeCalibration
#
#
#                # Reweighting #
#                wFail = op.extMethod("scalefactorWeightForFailingObject", returnType="double") # 
#                # double scalefactorWeightForFailingObject(double sf, double eff) {
#                #  return (1.-sf*eff)/(1.-eff);
#                #  }
#
#                # Method 1.a
#                # P(MC) = Π_{i tagged} eff_i Π_{j not tagged} (1-eff_j)
#                # P(data) = Π_{i tagged} SF_i x eff_i Π_{j not tagged} (1-SF_jxeff_j)
#                # w = P(data) / P(MC) = Π_{i tagged} SF_i Π_{j not tagged} (1-SF_jxeff_j)/(1-eff_j)
#                # NB : for  SF_i, self.DeepCsvSubjetMediumSF will find the correct SF based on the true flavour
#                #      however for eff_i, this comes from json SF and differs by flavour -> need multiSwitch
#                lambda_subjetWeight = lambda subjet : op.multiSwitch((subjet.nBHadrons>0,      # True bjets 
#                                                                      op.switch(self.lambda_subjetBtag(subjet),                                         # check if tagged
#                                                                                self.DeepCsvSubjetMediumSF(subjet),                                     # Tag : return SF_i
#                                                                                wFail(self.DeepCsvSubjetMediumSF(subjet),self.Ak8Eff_bjets(subjet)))),  # Not tagged : return (1-SF_jxeff_j)/(1-eff_j)
#                                                                     (subjet.nCHadrons>0,      # True cjets 
#                                                                      op.switch(self.lambda_subjetBtag(subjet),                                         # check if tagged
#                                                                                self.DeepCsvSubjetMediumSF(subjet),                                     # Tag : return SF_i
#                                                                                wFail(self.DeepCsvSubjetMediumSF(subjet),self.Ak8Eff_cjets(subjet)))),  # Not tagged : return (1-SF_jxeff_j)/(1-eff_j)
#                                                                      # Else : true lightjets 
#                                                                      op.switch(self.lambda_subjetBtag(subjet),                                               # check if tagged
#                                                                                self.DeepCsvSubjetMediumSF(subjet),                                           # Tag : return SF_i
#                                                                                wFail(self.DeepCsvSubjetMediumSF(subjet),self.Ak8Eff_lightjets(subjet))))     # Not tagged : return (1-SF_jxeff_j)/(1-eff_j)
#                self.ak8BtagReweighting = op.rng_product(self.ak8Jets, lambda j : lambda_subjetWeight(j.subJet1)*lambda_subjetWeight(j.subJet2))
#                sel = sel.refine("BtagAk8SF"+name , weight = [self.ak8BtagReweighting])
        
        return sel
       



    ###########################################################################
    #                            Lepton triggers                              #
    ###########################################################################
    def returnTriggers(self,keys):
        triggerRanges = returnTriggerRanges(self.era)
        # MC : just OR the different paths 
        if self.is_MC or self.isNanov7:
            return op.OR(*[trig for k in keys for trig in self.triggersPerPrimaryDataset[k]])
        else: # Only for NanoV6
            # Data : due to bug in NanoAOD production, check that the event run number is inside the ranges computed by bricalc
            conversion = {'SingleElectron':'1e','SingleMuon':'1mu','MuonEG':'1e1mu','DoubleEGamma':'2e','DoubleMuon':'2mu'}
            list_cond = []
            for trigKey in keys:
                rangekey = conversion[trigKey]
                rangeDict = {triggerRanges[rangekey][i]['name']:triggerRanges[rangekey][i]['runs'] for i in range(len(triggerRanges[rangekey]))}
                listTrig = self.triggersPerPrimaryDataset[trigKey]
                trigNames = [trig._parent.name for trig in listTrig]
                for trig,trigName in zip(listTrig,trigNames):
                    trigRanges = rangeDict[trigName]
                    list_cond.append(op.AND(trig,op.OR(*[op.in_range(r[0]-1,self.tree.run,r[1]+1) for r in trigRanges])))
                        # BEWARE : op.in_range is boundaries excluded !!! (hence extension via -1 and +1)
            return op.OR(*list_cond)

