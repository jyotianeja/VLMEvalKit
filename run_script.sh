#!/bin/bash

set -x

export SPLIT_THINK=True
echo $SPLIT_THINK

python run_evals_from_list.py --model v2-neel-p0-phi4mm-14b-triplemath-0202-r-extend-rlqnq-think --par 32 --ports 8002 

# python run_evals_from_list.py --model  v2-neel-p0-phi4mm-14b-triplemath-0202-r-extend-rlqnq-nothink --par 32 --ports 8002

python run_evals_from_list.py --model v2-neel-p0-phi4mm-14b-triplemath-0202-r-extend-rlqnq-hybrid --par 32 --ports 8005

python run_evals_from_list.py --model v2-neel-p0-phi4mm-14b-triplemath-0202-r-llwvr-think --par 32 --ports 8001

python run_evals_from_list.py --model v2-neel-p0-phi4mm-14b-triplemath-0202-r-llwvr-hybrid --par 32 --ports 8003


python run_evals_from_list.py --model neel-p0-phi4mm-14b-multi-16k-0217-r-rai-1e-6-f5sn8-think --par 32 --ports 8001

python run_evals_from_list.py --model neel-p0-phi4mm-14b-multi-16k-0217-r-rai-1e-6-f5sn8-hybrid --par 32 --ports 8002

python run_evals_from_list.py --model neel-p0-phi4mm-14b-multi-16k-0217-r-rai-5e-7-8b8bn-hybrid --par 32 --ports 8003

python run_evals_from_list.py --model neel-p0-phi4mm-14b-multi-16k-0217-r-rai-5e-7-8b8bn-think --par 32 --ports 8004

#######
python run_evals_from_list.py --model neel-p0-phi4mm-14b-multi-16k-0220-r-rai-6phvq-hybrid --par 32 --ports 8006

python run_evals_from_list.py --model neel-p0-phi4mm-14b-multi-16k-0220-r-rai-6phvq-think --par 32 --ports 8007

python run_evals_from_list.py --model neel-p0-phi4mm-14b-multi-16k-0220-r-rai-sm-pgmtq-hybrid  --par 32 --ports 8008

python run_evals_from_list.py --model neel-p0-phi4mm-14b-multi-16k-0220-r-rai-sm-pgmtq-think  --par 32 --ports 8009


python run_evals_from_list.py --model neel-p0-phi4mm-14b-multi-16k-0220-r-rai-nd-tjlnq-think --par 32 --ports 8001  

python run_evals_from_list.py --model neel-p0-phi4mm-14b-multi-16k-0220-r-rai-nd-tjlnq-hybrid --par 32 --ports 8002

==


python run_evals_from_list.py --model neel-p0-phi4mm-14b-m-16k-0220-r-rai-sm-nd-vl6dc-think --par 32 --ports 8003

python run_evals_from_list.py --model neel-p0-phi4mm-14b-m-16k-0220-r-rai-sm-nd-vl6dc-hybrid --par 32 --ports 8004


python run_evals_from_list.py --model neel-p0-phi4mm-14b-multi-16k-0220-r-rai-6phvq-hybrid-final --par 32 --ports 8025
python run_evals_from_list.py --model neel-p0-phi4mm-14b-multi-16k-0220-r-rai-6phvq-think-final --par 32 --ports 8026
python run_evals_from_list.py --model neel-p0-phi4mm-14b-multi-16k-0220-r-rai-6phvq-nothink-final --par 32 --ports 8027



mv *neel-p0-phi4mm-14b-multi-16k-0220-r-rai-6phvq-temp-0.2-hybrid_port8008* neel-p0-phi4mm-14b-multi-16k-0220-r-rai-6phvq-temp-0.2/8008/hybrid/
mv *neel-p0-phi4mm-14b-multi-16k-0220-r-rai-6phvq-temp-0.2-hybrid_port8007* neel-p0-phi4mm-14b-multi-16k-0220-r-rai-6phvq-temp-0.2/8007/hybrid/
mv *neel-p0-phi4mm-14b-multi-16k-0220-r-rai-6phvq-temp-0.2-hybrid_port8006* neel-p0-phi4mm-14b-multi-16k-0220-r-rai-6phvq-temp-0.2/8006/hybrid/
