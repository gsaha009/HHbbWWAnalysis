import yaml
import os


basepath = os.getcwd()
#basepath = '/home/ucl/cp3/kmondal/work/bamboodev/HHbbWWAnalysisSL/HHbbWWAnalysis'
outdir   = 'Plot_2017/results'              # change this name
config   = open('analysis2017_v7.yml','r')  # change this name 
outfile  = 'fail2017.log'                   # change this name

sucFiles =  [file.split('.')[0] for file in os.listdir(os.path.join(basepath,outdir))]
print('\nSuccessfully ran on : {}'.format(sucFiles))

confyml = yaml.safe_load(config)
samples = confyml.get('samples')
allFiles = [f for f in samples.keys()]
print('\nAll files in config : {}'.format(allFiles))

print('\nFailed ones : $$ ===>>>')
with open(outfile,'w') as outf: 
    for i in allFiles:
        if not i in sucFiles:
            print(i)
            outf.write(i+'\n')
