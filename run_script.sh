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

