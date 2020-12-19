#!/bin/bash

# boosted-El
#bambooRun -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o sync_Hbb0Wj_El --outputTreeName syncTree_hhbb1l_SR --Synchronization --Tight --Hbb0Wj --Channel El
#bambooRun -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o sync_Hbb1Wj_El --outputTreeName syncTree_hhbb1l_SR --Synchronization --Tight --Hbb1Wj --Channel El
#bambooRun -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o sync_Hbb2Wj_El --outputTreeName syncTree_hhbb1l_SR --Synchronization --Tight --Hbb2Wj --Channel El
bambooRun -t=40 --TTHIDLoose --NoSystematic -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o BstLoose_El --outputTreeName syncTree_hhbb1l_SR --Synchronization --Ak8 --Channel El
bambooRun -t=40 --TTHIDLoose --NoSystematic -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o BstLoose_Mu --outputTreeName syncTree_hhbb1l_SR --Synchronization --Ak8 --Channel Mu
#bambooRun -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o Mu_Res2b2W --Tight --Res2b2Wj --Channel El

# Resolved-El
#bambooRun -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o testsync_2b2Wj_El --outputTreeName syncTree_hhbb1l_SR --Synchronization --Tight --Res2b2Wj --Channel El
bambooRun -t=20 --TTHIDLoose -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o ResLoose_El --outputTreeName syncTree_hhbb1l_SR --Synchronization --Ak4 --Channel El
#bambooRun -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o testsync_2b1Wj_El --outputTreeName syncTree_hhbb1l_SR --Synchronization --Tight --Res2b1Wj --Channel El
bambooRun -t=20 --TTHIDLoose -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o ResLoose_Mu --outputTreeName syncTree_hhbb1l_SR --Synchronization --Ak4 --Channel Mu
#bambooRun -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o testsync_1b2Wj_El --outputTreeName syncTree_hhbb1l_SR --Synchronization --Tight --Res1b2Wj --Channel El
#bambooRun -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o sync_2b0Wj_El --outputTreeName syncTree_hhbb1l_SR --Synchronization --Tight --Res2b0Wj --Channel El
#bambooRun -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o sync_1b1Wj_El --outputTreeName syncTree_hhbb1l_SR --Synchronization --Tight --Res1b1Wj --Channel El
#bambooRun -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o sync_1b0Wj_El --outputTreeName syncTree_hhbb1l_SR --Synchronization --Tight --Res1b0Wj --Channel El
#bambooRun -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o sync_0b_El --outputTreeName syncTree_hhbb1l_SR --Synchronization --Tight --Res0b --Channel El

# Boosted-Mu
#bambooRun -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o sync_Hbb0Wj_Mu --outputTreeName syncTree_hhbb1l_SR --Synchronization --Tight --Hbb0Wj --Channel Mu
#bambooRun -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o sync_Hbb1Wj_Mu --outputTreeName syncTree_hhbb1l_SR --Synchronization --Tight --Hbb1Wj --Channel Mu
#bambooRun -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o sync_Hbb2Wj_Mu --outputTreeName syncTree_hhbb1l_SR --Synchronization --Tight --Hbb2Wj --Channel Mu

# Resolved-Mu
#bambooRun -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o sync_2b2Wj_Mu --outputTreeName syncTree_hhbb1l_SR --Synchronization --Tight --Res2b2Wj --Channel Mu
#bambooRun -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o sync_2b1Wj_Mu --outputTreeName syncTree_hhbb1l_SR --Synchronization --Tight --Res2b1Wj --Channel Mu
#bambooRun -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o sync_1b2Wj_Mu --outputTreeName syncTree_hhbb1l_SR --Synchronization --Tight --Res1b2Wj --Channel Mu
#bambooRun -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o sync_2b0Wj_Mu --outputTreeName syncTree_hhbb1l_SR --Synchronization --Tight --Res2b0Wj --Channel Mu
#bambooRun -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o sync_1b1Wj_Mu --outputTreeName syncTree_hhbb1l_SR --Synchronization --Tight --Res1b1Wj --Channel Mu
#bambooRun -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o sync_1b0Wj_Mu --outputTreeName syncTree_hhbb1l_SR --Synchronization --Tight --Res1b0Wj --Channel Mu
#bambooRun -m SkimmerHHtobbWWSL.py:SkimmerNanoHHtobbWWSL analysis2016_v6_synchro_HH.yml -o sync_0b_Mu --outputTreeName syncTree_hhbb1l_SR --Synchronization --Tight --Res0b --Channel Mu

