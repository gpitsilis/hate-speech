#!/bin/bash

echo $1

for i in `cat $1` 
do
 echo "sorting "$i
 cat $i | sort > S$i
 rm $i
 mv S$i $i
done
