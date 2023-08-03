#!/bin/bash

# re means respiration, _1 means 0.1 sparsity
python3 sra_train.py --cuda --data=re_1
python3 sra_train.py --cuda --data=re_2
python3 sra_train.py --cuda --data=re_3
python3 sra_train.py --cuda --data=re_4
python3 sra_train.py --cuda --data=re_5

# ge means gesture, _1 means 0.1 sparsity
python3 sra_train.py --cuda --data=ge_1
python3 sra_train.py --cuda --data=ge_2
python3 sra_train.py --cuda --data=ge_3
python3 sra_train.py --cuda --data=ge_4
python3 sra_train.py --cuda --data=ge_5

# ac means gesture, _1 means 0.1 sparsity
python3 sra_train.py --cuda --data=ac_1
python3 sra_train.py --cuda --data=ac_2
python3 sra_train.py --cuda --data=ac_3
python3 sra_train.py --cuda --data=ac_4
python3 sra_train.py --cuda --data=ac_5
