#!/bin/bash

# ============ basic configuration ============
MG5_PATH=/afs/cern.ch/work/f/fdibello/ChrisMalte/MG5_aMC_v3_5_7
DELPHES_PATH=/afs/cern.ch/work/f/fdibello/ChrisMalte/delphes
OUTPUT_PATH=/afs/cern.ch/work/f/fdibello/ChrisMalte/output

## some env variables are required by the softwares
#LHAPDFCONFIG=/cvmfs/sft.cern.ch/lcg/releases/MCGenerators/lhapdf/6.5.3-3fa11/x86_64-el9-gcc12-opt/bin/lhapdf-config
#LHAPDF_DATA_PATH=/cvmfs/sft.cern.ch/lcg/releases/MCGenerators/lhapdf/6.5.3-3fa11/x86_64-el9-gcc12-opt/share/LHAPDF
LHAPDFCONFIG=/afs/cern.ch/work/f/fdibello/ChrisMalte/MG5_aMC_v3_5_7/HEPTools/lhapdf6_py3/bin/lhapdf-config
LHAPDF_DATA_PATH=/afs/cern.ch/work/f/fdibello/ChrisMalte/MG5_aMC_v3_5_7/HEPTools/lhapdf6_py3/share/LHAPDF
PYTHIA8DATA=/afs/cern.ch/work/f/fdibello/ChrisMalte/MG5_aMC_v3_5_7/HEPTools/pythia8/share/Pythia8/xmldoc

#DELPHES_CARD_PATH=/afs/cern.ch/work/f/fdibello/ChrisMalte/jetclass1/jetclass_generation/delphes_card.tcl
#DELPHES_CARD_PATH=/afs/cern.ch/work/f/fdibello/ChrisMalte/jetclass1/jetclass_generation/delphes_card_smeared.tcl
DELPHES_CARD_PATH=/afs/cern.ch/work/f/fdibello/ChrisMalte/jetclass1/jetclass_generation/delphes_card_correctM.tcl

TAG=${1}

#Folder="/eos/project/a/atlas-vhbb-ml/boosted_1L/inputs/mcade21May2020/notSummed"
scp -r /eos/user/f/fdibello/BigBello/py8* . 


sed -i "s/^Random:seed = 0/Random:seed = ${TAG}/" py8.dat

g++ py8_main.cc -o py8_main -I$MG5_PATH/HEPTools/pythia8//include -ldl -fPIC -lstdc++ -std=c++11 -O2 -DHEPMC2HACK -DGZIP -I$MG5_PATH/HEPTools/zlib/include -L$MG5_PATH/HEPTools/zlib/lib -Wl,-rpath,$MG5_PATH/HEPTools/zlib/lib -lz -L$MG5_PATH/HEPTools/pythia8//lib -Wl,-rpath,$MG5_PATH/HEPTools/pythia8//lib -lpythia8 -ldl -I$MG5_PATH/HEPTools/hepmc/include -L$MG5_PATH/HEPTools/hepmc/lib -Wl,-rpath,$MG5_PATH/HEPTools/hepmc/lib -lHepMC -DHEPMC2

./py8_main


mv events.hepmc events_${TAG}.hepmc

source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh

$DELPHES_PATH/DelphesHepMC2 $DELPHES_CARD_PATH events_delphes_${TAG}.root events_${TAG}.hepmc

ls -l
echo '-> Workspace created..'


#cp  events_${TAG}.hepmc /eos/user/f/fdibello/BigBello/data_py/events_${TAG}.hepmc
cp  events_delphes_${TAG}.root /eos/user/f/fdibello/BigBello/data_py_top/events_delphes_${TAG}.root


ls -l /eos/user/f/fdibello/BigBello/data_py_top/

cd /afs/cern.ch/work/f/fdibello/ChrisMalte/delphes/

root -l -b -q "makeNtuples.C(\"/eos/user/f/fdibello/BigBello/data_py_top/events_delphes_${TAG}.root\",\"/eos/user/f/fdibello/BigBello/data_py_top/output_Dijetcc_smeared_${TAG}.root\")" 
#root -l -b -q "makeNtuples.C(\"/eos/user/f/fdibello/BigBello/data_py_top/events_delphes_${TAG}.root\",\"/eos/user/f/fdibello/BigBello/data_py_top/output_Zj_PSmurr2_NOsmeared_${TAG}.root\")" 

cd -

rm -rf /eos/user/f/fdibello/BigBello/cluster/${TAG}/ 
rm /eos/user/f/fdibello/BigBello/data_py_top/events_delphes_${TAG}.root

echo '-> end!'
