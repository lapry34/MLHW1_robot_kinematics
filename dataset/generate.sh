#! /bin/bash

python run.py -env r2 -seed 1979543 -steps 100000 --log > logs/logfile_r2.csv
python run.py -env r3 -seed 1979543 -steps 100000 --log > logs/logfile_r3.csv
python run.py -env r5 -seed 1979543 -steps 100000 --log > logs/logfile_r5.csv

python run.py -env r2 -seed 1982783 -steps 100000 --log > validation/r2.csv
python run.py -env r3 -seed 1982783 -steps 100000 --log > validation/r3.csv
python run.py -env r5 -seed 1982783 -steps 100000 --log > validation/r5.csv