#! /bin/env python

import math
import argparse
import ROOT

def format_eta_bin(eta_bin):
    return 'ptabseta<%.1f' % (eta_bin[1]) if (eta_bin[0] == 0) else 'ptabseta%.1f-%.1f' % (eta_bin[0], eta_bin[1])

parser = argparse.ArgumentParser()
parser.add_argument('file', help='ROOT file containing electron scale factors')
parser.add_argument('-s', '--suffix', help='Suffix to append at the end of the output filename', required=True)

args = parser.parse_args()


f = ROOT.TFile.Open(args.file)

h = f.Get("EGamma_SF2D")
hu = f.Get("EGamma_SF2D_Up")
hd = f.Get("EGamma_SF2D_Down")

# Get binning
eta_binning = []
for i in range(1, h.GetXaxis().GetNbins() + 1):
    if len(eta_binning) == 0:
        eta_binning.append(h.GetXaxis().GetBinLowEdge(i))
        eta_binning.append(h.GetXaxis().GetBinUpEdge(i))
    else:
        eta_binning.append(h.GetXaxis().GetBinUpEdge(i))

pt_binning = []
for i in range(1, h.GetYaxis().GetNbins() + 1):
    if len(pt_binning) == 0:
        pt_binning.append(h.GetYaxis().GetBinLowEdge(i))
        pt_binning.append(h.GetYaxis().GetBinUpEdge(i))
    else:
        pt_binning.append(h.GetYaxis().GetBinUpEdge(i))

eta = 'Eta' if eta_binning[0] < 0 else 'AbsEta'
json_content = {'dimension': 2, 'variables': [eta, 'Pt'], 'binning': {'x': eta_binning, 'y': pt_binning}, 'data': [], 'error_type': 'absolute'}
json_content_data = json_content['data']

for i in range(0, len(eta_binning) - 1):
    eta_data = {'bin': [eta_binning[i], eta_binning[i + 1]], 'values': []}
    for j in range(0, len(pt_binning) - 1):
        pt_data = {'bin': [pt_binning[j], pt_binning[j + 1]], 'value': h.GetBinContent(i+1,j+1), 'error_low': h.GetBinContent(i+1,j+1)-hd.GetBinContent(i+1,j+1), 'error_high': hu.GetBinContent(i+1,j+1)-h.GetBinContent(i+1,j+1)}
        eta_data['values'].append(pt_data)

    json_content_data.append(eta_data)

# Save JSON file
filename = 'TTHSF_%s.json' % (args.suffix)
with open(filename, 'w') as j:
    import json
    json.dump(json_content, j, indent=2)
