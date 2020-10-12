import os
import sys
from copy import copy, deepcopy
from bamboo import treefunctions as op
from highlevelLambdas import *

# ------------------------------------------------------------------------------------------------------------------------------------------------------ #
#                                                                      BDT - Tallin                                                                      #
# ------------------------------------------------------------------------------------------------------------------------------------------------------ #
# ----- Whad tagger ----- #
def applyBDTforWhadTagger(lepton, jets, bjets, lightJets, model_even, model_odd, event):
    invars = [ lightJets[0].p4.Pt(),
               lightJets[0].btagCSVV2,
               lightJets[0].qgl,
               lightJets[1].p4.Pt(),
               lightJets[1].qgl,
               op.deltaR((lightJets[0].p4+lightJets[1].p4),lepton.p4),
               op.deltaR(lightJets[0].p4, lightJets[1].p4),
               op.invariant_mass(lightJets[0].p4, lightJets[1].p4),
               op.rng_len(bjets),
               op.rng_len(jets)
           ]

    output = op.switch(event%2,model_odd(*invars),model_even(*invars))
    return output[0]

# ------ BDTfullRecoResolved ------ #
def evaluateBDTfullRecoResolved(fakeLepColl,lep,met,jets,bJets,j1,j2,j3,j4,model_even,model_odd,event,HLL):
    #inputs = [op.c_float(0.)]*21
    invars = [HLL.mindr_lep1_jet(lep,jets),                                                                                         # mindr_lep1_jet
              op.invariant_mass(HLL.bJetCorrP4(j1), HLL.bJetCorrP4(j2)),                                                # m_Hbb_regCorr 
              HLL.HHP4_simple_met(HLL.bJetCorrP4(j1)+HLL.bJetCorrP4(j2), j3.p4, j4.p4, lep.p4, met.p4).M(),             # mHH_simple_met 
              HLL.Wlep_met_simple(lep.p4, met.p4).M(),                                                                              # mWlep_met_simple
              HLL.HWW_met_simple(j3.p4,j4.p4,lep.p4,met.p4).M(),                                                                    # mWW_simple_met
              HLL.Wjj_simple(j3.p4, j4.p4).M(),                                                                                     # mWjj_simple
              HLL.comp_cosThetaS(HLL.bJetCorrP4(j1),HLL.bJetCorrP4(j2)),                                                # cosThetaS_Hbb                   
              HLL.comp_cosThetaS(j3.p4, j4.p4),                                                                                     # cosThetaS_Wjj_simple     
              HLL.comp_cosThetaS(HLL.Wjj_simple(j3.p4,j4.p4),HLL.Wlep_met_simple(lep.p4,met.p4)),                                   # cosThetaS_WW_simple_met  
              HLL.comp_cosThetaS(HLL.bJetCorrP4(j1)+HLL.bJetCorrP4(j2), HLL.HWW_met_simple(j3.p4,j4.p4,lep.p4,met.p4)), # cosThetaS_HH_simple_met     
              op.rng_len(jets),                                                                                                     # nJet
              op.rng_len(bJets),                                                                                                    # nBJetMedium
              op.deltaR(j1.p4, lep.p4),                                                                                       # dR_b1lep
              op.deltaR(j2.p4, lep.p4),                                                                                       # dR_b2lep
              HLL.lambdaConePt(lep),                                                                                                # lep_conePt
              HLL.bJetCorrP4(j1).Pt(),                                                                                        # selJet1_Hbb_pT
              HLL.bJetCorrP4(j2).Pt(),                                                                                        # selJet2_Hbb_pT 
              HLL.MET_LD(met,jets,fakeLepColl),                                                                                     # met_LD
              HLL.HTfull(fakeLepColl,HLL.bJetCorrP4(j1),HLL.bJetCorrP4(j2),j3.p4,j4.p4),                                # HT
              op.min(HLL.mT2(HLL.bJetCorrP4(j1),lep.p4 ,met.p4), HLL.mT2(HLL.bJetCorrP4(j2), lep.p4, met.p4)),          # mT_top_3particle
              HLL.MT(lep,met)                                                                                                       # mT_W
          ]

    output = op.switch(event%2,model_odd(*invars),model_even(*invars))
    return output[0]

# ------ BDTmissRecoResolved ------ #
def evaluateBDTmissRecoResolved(fakeLepColl,lep,met,jets,bJets,j1,j2,j3,j4,model_even,model_odd,event,HLL):
    #inputs = [op.c_float(0.)]*21
    invars = [op.min(HLL.mT2(HLL.bJetCorrP4(j1),lep.p4 ,met.p4), HLL.mT2(HLL.bJetCorrP4(j2), lep.p4, met.p4)),# mT_top_3particle
              HLL.MT(lep,met),                                                                        # mT_W
              HLL.mindr_lep1_jet(lep,jets),                                                           # mindr_lep1_jet
              op.deltaR(j1.p4, lep.p4),                                                               # dR_b1lep
              op.deltaR(j2.p4, lep.p4),                                                               # dR_b2lep             
              op.invariant_mass(HLL.bJetCorrP4(j1), HLL.bJetCorrP4(j2)),                              # m_Hbb_regCorr 
              HLL.bJetCorrP4(j1).Pt(),                                                                # selJet1_Hbb_pT
              HLL.bJetCorrP4(j2).Pt(),                                                                # selJet2_Hbb_pT 
              op.deltaR(j3.p4, HLL.Wlep_simple(j1.p4,j2.p4,lep.p4,met)),                              # dr_Wj1_lep_simple
              op.rng_len(bJets),                                                                      # nBJetMedium
              HLL.lambdaConePt(lep),                                                                  # lep_conePt
              HLL.MET_LD(met,jets,fakeLepColl),                                                       # met_LD
              HLL.HTmiss(fakeLepColl,HLL.bJetCorrP4(j1),HLL.bJetCorrP4(j2),j3.p4)                     # HT
          ]

    output = op.switch(event%2,model_odd(*invars),model_even(*invars))
    return output[0]



# ------ BDTfullRecoBoosted ------ #
def evaluateBDTfullRecoBoosted(fakeLepColl,lep,met,jets,bJets,j1,j2,j3,j4,model_even,model_odd,nMedBJets,event,HLL):
    #inputs = [op.c_float(0.)]*21
    invars = [HLL.mindr_lep1_jet(lep,jets),                                                                  # mindr_lep1_jet
              op.invariant_mass(j1.p4, j2.p4),                                                               # m_Hbb_regCorr 
              HLL.HHP4_simple_met(j1.p4+j2.p4, j3.p4, j4.p4, lep.p4, met.p4).M(),                            # mHH_simple_met 
              HLL.Wlep_met_simple(lep.p4, met.p4).M(),                                                       # mWlep_met_simple
              HLL.HWW_met_simple(j3.p4,j4.p4,lep.p4,met.p4).M(),                                             # mWW_simple_met
              HLL.Wjj_simple(j3.p4, j4.p4).M(),                                                              # mWjj_simple
              HLL.comp_cosThetaS(j1.p4, j2.p4),                                                              # cosThetaS_Hbb                   
              HLL.comp_cosThetaS(j3.p4, j4.p4),                                                                          # cosThetaS_Wjj_simple     
              HLL.comp_cosThetaS(HLL.Wjj_simple(j3.p4,j4.p4),HLL.Wlep_met_simple(lep.p4,met.p4)),                        # cosThetaS_WW_simple_met  
              HLL.comp_cosThetaS(j1.p4+j2.p4, HLL.HWW_met_simple(j3.p4,j4.p4,lep.p4,met.p4)),         # cosThetaS_HH_simple_met     
              op.rng_len(jets),                                                                       # nJet
              #op.rng_len(bJets),                                                                      # nBJetMedium
              nMedBJets,
              op.deltaR(j1.p4, lep.p4),                                                               # dR_b1lep
              op.deltaR(j2.p4, lep.p4),                                                               # dR_b2lep
              HLL.lambdaConePt(lep),                                                                  # lep_conePt
              j1.pt,                                                                                  # selJet1_Hbb_pT
              j2.pt,                                                                                  # selJet2_Hbb_pT 
              HLL.MET_LD(met,jets,fakeLepColl),                                                       # met_LD
              HLL.HTfull(fakeLepColl,j1.p4,j2.p4,j3.p4,j4.p4),                                        # HT
              op.min(HLL.mT2(j1.p4, lep.p4 ,met.p4), HLL.mT2(j2.p4, lep.p4, met.p4)),                 # mT_top_3particle
              HLL.MT(lep,met)                                                                         # mT_W
          ]

    output = op.switch(event%2,model_odd(*invars),model_even(*invars))
    return output[0]


# ------ BDTmissRecoBoosted ------ #
def evaluateBDTmissRecoBoosted(fakeLepColl,lep,met,jets,bJets,j1,j2,j3,j4,model_even,model_odd,nMedBJets,event,HLL):
    #inputs = [op.c_float(0.)]*21
    inputs = [op.min(HLL.mT2(j1.p4, lep.p4 ,met.p4), HLL.mT2(j2.p4, lep.p4, met.p4)),                 # mT_top_3particle
              HLL.MT(lep,met),                                                                        # mT_W
              HLL.mindr_lep1_jet(lep,jets),                                                           # mindr_lep1_jet
              op.deltaR(j1.p4, lep.p4),                                                               # dR_b1lep
              op.deltaR(j2.p4, lep.p4),                                                               # dR_b2lep             
              op.invariant_mass(j1.p4, j2.p4),                                                        # m_Hbb_regCorr 
              j1.pt,                                                                                  # selJet1_Hbb_pT
              j2.pt,                                                                                  # selJet2_Hbb_pT 
              op.deltaR(j3.p4, HLL.Wlep_simple(j1.p4,j2.p4,lep.p4,met)),                              # dr_Wj1_lep_simple
              nMedBJets,
              #op.rng_len(bJets),                                                                      # nBJetMedium
              HLL.lambdaConePt(lep),                                                                  # lep_conePt
              HLL.MET_LD(met,jets,fakeLepColl),                                                       # met_LD
              HLL.HTmiss(fakeLepColl,j1.p4,j2.p4,j3.p4)                                               # HT
          ]

    output = op.switch(event%2,model_odd(*invars),model_even(*invars))
    return output[0]

# ------------------------------------------------------------------------------------------------------------------------------------------------------ #
#                                                                         DNN - UCL                                                                      #
# ------------------------------------------------------------------------------------------------------------------------------------------------------ #
def evaluateDNNfullRecoResolved(lep,fakelepcoll,met,jets,bJets,j3,j4,model1,model2,model3,model4,model5,event,HLL):
    invars = [op.static_cast("UInt_t",op.rng_len(jets)), 
              op.static_cast("UInt_t",op.rng_len(bJets)),
              lep.pt,
              op.abs(HLL.SinglepMet_dPhi(lep,met)),
              HLL.SinglepMet_Pt(lep,met),
              HLL.MT(lep,met),
              HLL.MET_LD(met, jets, fakelepcoll),
              bJets[0].pt, 
              j3.pt, 
              op.deltaR(bJets[0].p4, bJets[1].p4),
              op.deltaR(bJets[1].p4,j3.p4), 
              op.deltaR(j3.p4,j4.p4),
              op.invariant_mass(HLL.bJetCorrP4(bJets[0]),HLL.bJetCorrP4(bJets[1])), 
              op.invariant_mass(j3.p4, j4.p4),
              op.abs(HLL.SinglepMet_dPhi(bJets[0], met)), 
              op.abs(HLL.SinglepMet_dPhi(j3, met)), 
              op.deltaR(bJets[0].p4, lep.p4), 
              op.deltaR(j3.p4, lep.p4),
              HLL.MinDiJetDRTight(bJets[0], bJets[1], j3, j4),
              HLL.mindr_lep1_jet(lep, jets),
              HLL.MT_W1W2_ljj(lep,j3,j4,met),
              HLL.HT2R_l4jmet(lep,bJets[0],bJets[1],j3,j4,met),
              op.min(HLL.mT2(HLL.bJetCorrP4(bJets[0]),lep.p4 ,met.p4), HLL.mT2(HLL.bJetCorrP4(bJets[1]), lep.p4, met.p4)),
              HLL.HWW_simple(j3.p4,j4.p4,lep.p4,met).M(),
              HLL.dR_Hww(j3.p4,j4.p4,lep.p4,met)
          ]
    
    output = op.multiSwitch((event%5 == 0 ,model1(*invars)),
                            (event%5 == 1 ,model2(*invars)),
                            (event%5 == 2 ,model3(*invars)),
                            (event%5 == 3 ,model4(*invars)),
                            model5(*invars))
    return output
