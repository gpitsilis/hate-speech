#!/bin/bash

to=15
from=1

rm all_unigrams_relu.csv
rm all_unigrams_RS_relu.csv
rm all_unigrams_NS_relu.csv
rm all_unigrams_NR_relu.csv
rm all_unigrams_NRS_relu.csv

for i in `seq $from $to` 
do
 echo $i
 
 cat ${i}_unigrams_relu.csv >> all_unigrams_relu.csv
 cat ${i}_unigrams_RS_relu.csv >> all_unigrams_RS_relu.csv
 cat ${i}_unigrams_NS_relu.csv >> all_unigrams_NS_relu.csv
 cat ${i}_unigrams_NR_relu.csv >> all_unigrams_NR_relu.csv
 cat ${i}_unigrams_NRS_relu.csv >> all_unigrams_NRS_relu.csv

done

python merging_5.py
