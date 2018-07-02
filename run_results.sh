#!/bin/sh

rm O.csv
rm NR.csv
rm NS.csv
rm RS.csv
rm NRS.csv

for i in `seq 1 15` 
do

 echo $i
 cat ${i}_unigrams_relu.csv     >> O.csv
 cat ${i}_unigrams_NR_relu.csv  >> NR.csv
 cat ${i}_unigrams_NS_relu.csv  >> NS.csv
 cat ${i}_unigrams_RS_relu.csv  >> RS.csv
 cat ${i}_unigrams_NRS_relu.csv >> NRS.csv

done


python validate.py O.csv
python validate.py NR.csv
python validate.py NS.csv
python validate.py RS.csv
python validate.py NRS.csv


echo "deleting temp files"
rm O.csv
rm NR.csv
rm NS.csv
rm RS.csv
rm NRS.csv
