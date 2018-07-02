to=15
from=1

for var in `seq $from $to`
do
  echo $var
  python classifier.py NRS $var
done
