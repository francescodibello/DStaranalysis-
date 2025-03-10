1. per analizzare mini-ntuple root -l -b analyse.C
2. Mini ntupla è prodotta da una macro-ntuple di delphes. Il dumper è nel file MakeNtuples.C
Per girare bisogna scaricare delphes come spiegato qui: https://github.com/jet-universe/jetclass2_generation
Di fatto basta scaricare delphes, setupparlo su lxplus con:

source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh

e poi girare:

root -l -b -q "makeNtuples.C(\"/eos/user/f/fdibello/BigBello/events_delphes.root\",
\"/eos/user/f/fdibello/BigBello/delphes/test.root\")"

3. il transformer è nella cartella "ML"

