#!/bin/sh

./sort.sh list_O.csv
./sort.sh list_NRS.csv
./sort.sh list_NS.csv
./sort.sh list_NR.csv
./sort.sh list_RS.csv 

python mix.py list_O.csv list_NRS.csv list_NR.csv mixfile_O_NRS_NR.csv
python validate_mix.py mixfile_O_NRS_NR.csv
rm mixfile_O_NRS_NR.csv

python mix.py list_O.csv list_NRS.csv list_NS.csv mixfile_O_NRS_NS.csv
python validate_mix.py mixfile_O_NRS_NS.csv
rm mixfile_O_NRS_NS.csv

python mix.py list_O.csv list_NRS.csv list_RS.csv mixfile_O_NRS_RS.csv
python validate_mix.py mixfile_O_NRS_RS.csv
rm mixfile_O_NRS_RS.csv

python mix.py list_O.csv list_NS.csv list_RS.csv mixfile_O_NS_RS.csv
python validate_mix.py mixfile_O_NS_RS.csv
rm mixfile_O_NS_RS.csv

python mix.py list_O.csv list_NS.csv list_NR.csv mixfile_O_NS_NR.csv
python validate_mix.py mixfile_O_NS_NR.csv
rm mixfile_O_NS_NR.csv

python mix.py list_O.csv list_RS.csv list_NR.csv mixfile_O_RS_NR.csv
python validate_mix.py mixfile_O_RS_NR.csv
rm mixfile_O_RS_NR.csv

python mix.py list_NRS.csv list_RS.csv list_NR.csv mixfile_NRS_RS_NR.csv
python validate_mix.py mixfile_NRS_RS_NR.csv
rm mixfile_NRS_RS_NR.csv

python mix.py list_NRS.csv list_NR.csv list_NS.csv mixfile_NRS_NR_NS.csv
python validate_mix.py mixfile_NRS_NR_NS.csv
rm mixfile_NRS_NR_NS.csv

python mix.py list_NRS.csv list_RS.csv list_NS.csv mixfile_NRS_RS_NS.csv
python validate_mix.py mixfile_NRS_RS_NS.csv
rm mixfile_NRS_RS_NS.csv

python mix.py list_NR.csv list_RS.csv list_NS.csv mixfile_NR_RS_NS.csv
python validate_mix.py mixfile_NR_RS_NS.csv
rm mixfile_NR_RS_NS.csv


