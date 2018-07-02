# for single classifier

from statistics import mode, StatisticsError
import numpy as np
import sys
import math

import random

cmd_args = len(sys.argv)

if (cmd_args<2):
  print "No filename given."
  sys.exit(0)
else:
  if (cmd_args==3):
     num_of_recs = int(sys.argv[2])
  else:
     num_of_recs = 0

  input_filename = sys.argv[1]
  print "evaluating file:", input_filename, " for ", (num_of_recs * 15998) , "(0 for all) records"

TP_N = 0
TP_R = 0
TP_S = 0

FP_N = 0
FP_R = 0
FP_S = 0

FN_N = 0
FN_R = 0
FN_S = 0

lineCounter = 0

counterN = 0
counterR = 0
counterS = 0


default_classifier = 0


inpt1 = open(input_filename, "r+")

line1 = inpt1.readline()

rec_counter = 0

uncertainty = 0

while (line1 != ''):

    rec_counter = rec_counter + 1
    if ((rec_counter > (num_of_recs*15998)) and ((num_of_recs*15998) > 0)):
       print "Premature exit"
       break

    if (line1 == '\n'):

        print "run out of records."
        sys.exit(0)

    fields1 = line1.split(",")

    lineCounter += 1

    tid = fields1[0]

    neutral_1 = float(fields1[2])
    racism_1  = float(fields1[3])
    sexism_1  = float(fields1[4])

    # compute the standard deviation of these 3 values. (uncertainty)
    avg = (neutral_1 + racism_1 + sexism_1)/3.0
  
    std_NRS = math.sqrt((math.pow((neutral_1 - avg),2) + math.pow((racism_1 - avg),2) + math.pow((sexism_1 - avg),2)) / 3)   

    uncertainty = uncertainty + std_NRS

    label = fields1[1]
    # compute the f-score and compare the result with 1st classifier alone

    lst_1 = [ neutral_1, racism_1, sexism_1 ]

    max_index = np.empty([3], dtype=int)

    max_value = max(lst_1)
    idx = max_index[0] = lst_1.index(max_value)

    if (label=='0.0'):
        counterN += 1.0
        if (float(idx) == float(label)):
            TP_N += 1.0

        if (idx != 0):
            FN_N += 1.0

    if ((idx == 0) and (str(label) != "0.0")):
        FP_N += 1.0


    if (label=='1.0'):
        counterR += 1.0
        if (float(idx) == float(label)):
            TP_R += 1.0

        if (idx != 1):
            FN_R += 1.0

    if ((idx == 1) and (str(label) != "1.0")):
        FP_R += 1.0


    if (label=='2.0'):
        counterS += 1.0
        if (float(idx) == float(label)):
            TP_S += 1.0

        if (idx != 2):
            FN_S += 1.0

    if ((idx== 2) and (str(label) != "2.0")):
        FP_S += 1.0

    # compute final label here
    # re-calculate the F-score here

    line1 = inpt1.readline()


uncertainty = float(uncertainty)/float(lineCounter)
output_file = input_filename + "_result.out"
f = open(output_file, 'w')

f.write('\n' + "Performance of Classifier:")
f.write('\n' + "TP,FP,FN:")
f.write('\n' + str(TP_N) + " " + str(FP_N) + " " + str(FN_N))
f.write('\n' + str(TP_R) + " " + str(FP_R) + " " + str(FN_R))
f.write('\n' + str(TP_S) + " " + str(FP_S) + " " + str(FN_S))

f.write('\n' + "avg. std. dev:" + str(uncertainty))
PN = TP_N / (TP_N + FP_N)
PR = TP_R / (TP_R + FP_R)
PS = TP_S / (TP_S + FP_S)

RN = TP_N / (TP_N + FN_N)
RR = TP_R / (TP_R + FN_R)
RS = TP_S / (TP_S + FN_S)

FN = 2 * PN * RN / (PN + RN)
FR = 2 * PR * RR / (PR + RR)
FS = 2 * PS * RS / (PS + RS)

f.write('\n' + "Default model")
f.write('\n' + "--------------")
f.write('\n' + "Results over:" + str(lineCounter))
f.write('\n' + "P neutral, P racism, P sexism:" + str(PN) +" "+ str(PR) + " " + str(PS))
f.write('\n' + "R neutral, R racism, R sexism:" + str(RN) +" "+ str(RR) + " " + str(RS))
f.write('\n' + "F Neutral:" + str(FN)+ " F Racism:" + str(FR)+ " F Sexism:" + str(FS))

F = (FN * counterN + FR * counterR+ FS * counterS) / ( counterN + counterR + counterS )
f.write('\n' + "Combined F:" + str(F) + " F_neutral:" + str(FN) + " F_racism:" +str(FR) + " F_sexism:" + str(FS))
f.write('\n' + "Combined P:" + str((PN * counterN + PR * counterR+ PS * counterS) / ( counterN + counterR + counterS )))
f.write('\n' + "Combined R:" + str((RN * counterN + RR * counterR+ RS * counterS) / ( counterN + counterR + counterS )))
f.close()
