# validate ensemble of 3 classifiers

from statistics import mode, StatisticsError
import numpy as np
import sys

import random


cmd_args = len(sys.argv)


print cmd_args

if (cmd_args<2):
  print "No input filename given."
  sys.exit(0)
else:
  if (cmd_args==3):
     num_of_recs = int(sys.argv[2])
  else:
     num_of_recs = 0

  print "evaluating for ", (num_of_recs * 15998) , "(0 for all) records"


input_file = sys.argv[1]


TP_N = 0
TP_R = 0
TP_S = 0

FP_N = 0
FP_R = 0
FP_S = 0

FN_N = 0
FN_R = 0
FN_S = 0

TP_Nd = 0
TP_Rd = 0
TP_Sd = 0

FP_Nd = 0
FP_Rd = 0
FP_Sd = 0

FN_Nd = 0
FN_Rd = 0
FN_Sd = 0

NN_ = 0
SS_ = 0
RR_ = 0
NS_ = 0
NR_ = 0
RN_ = 0
RS_ = 0
SN_ = 0
SR_ = 0

lineCounter = 0

counterN = 0
counterR = 0
counterS = 0

noconsensus = 0
re_labels = 0
ok_re_labels = 0

default_classifier = 0


inpt1 = open(input_file, "r+")

line1 = inpt1.readline()


rec_counter = 0

while (line1 != ''):


    if ((rec_counter % 100000) == 0):
       print rec_counter/100000, "00 k "

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
    if (not (tid == fields1[5] == fields1[10])):
       print "Oooops. Inconsistent input files at line:", lineCounter
       break
       sys.exit(0)

    if (not (fields1[1] == fields1[6] == fields1[11])):
       print "Oooops. Inconsistent labes in input files at line:", lineCounter
       sys.exit(0)

    neutral_1 = float(fields1[2])
    neutral_2 = float(fields1[7])
    neutral_3 = float(fields1[12])

    racism_1 = float(fields1[3])
    racism_2 = float(fields1[8])
    racism_3 = float(fields1[13])

    sexism_1 = float(fields1[4])
    sexism_2 = float(fields1[9])
    sexism_3 = float(fields1[14])

    label = fields1[1]
    # compute f-score here and compare the result with 1st classifier alone

    lst_1 = [ neutral_1, racism_1, sexism_1 ]
    lst_2 = [ neutral_2, racism_2, sexism_2 ]
    lst_3 = [ neutral_3, racism_3, sexism_3 ]

    max_index = np.empty([3], dtype=int)


    max_value = max(lst_1)
    max_index[0] = lst_1.index(max_value)

    max_value = max(lst_2)
    max_index[1] = lst_2.index(max_value)

    max_value = max(lst_3)
    max_index[2] = lst_3.index(max_value)


    if (label=='0.0'):
        counterN += 1.0
        if (float(max_index[0]) == float(label)):
            TP_N += 1.0

        if (max_index[0] != 0):
            FN_N += 1.0

    if ((max_index[0] == 0) and (str(label) != "0.0")):
        FP_N += 1.0



    if (label=='1.0'):
        counterR += 1.0
        if (float(max_index[0]) == float(label)):
            TP_R += 1.0

        if (max_index[0] != 1):
            FN_R += 1.0

    if ((max_index[0] == 1) and (str(label) != "1.0")):
        FP_R += 1.0



    if (label=='2.0'):
        counterS += 1.0
        if (float(max_index[0]) == float(label)):
            TP_S += 1.0

        if (max_index[0] != 2):
            FN_S += 1.0

    if ((max_index[0] == 2) and (str(label) != "2.0")):
        FP_S += 1.0

    # apply conditional voting here

    final_decision = max_index[0]

    if not (max_index[0] == max_index[1] == max_index[2]):

            noconsensus += 1

            # averaging

            s_neutral = neutral_1 + neutral_2 + neutral_3
            s_racism  = racism_1  + racism_2  + racism_3
            s_sexism  = sexism_1  + sexism_2  + sexism_3

            values = [s_neutral, s_racism, s_sexism]
            average_decision = np.argmax(values)

            m1 = max(lst_1)
            m2 = max(lst_2)
            m3 = max(lst_3)

            most_conf_classifier = np.argmax([m1, m2, m3]) 

            strongest_decision = max_index[most_conf_classifier]

            try:
                voting_decision = mode([max_index[0], max_index[1], max_index[2]]) 

                # counting re-labelings
                if (max_index[0] != final_decision):
                    re_labels += 1
                    # counting succesful re-labelings
                    if (float(max_index[0]) == float(label)):
                        ok_re_labels += 1

                final_decision = voting_decision

                if (voting_decision != strongest_decision):
                    lab = int(float(label))


                    if (lab not in [voting_decision, strongest_decision]):
                       rrr = 1 

            except StatisticsError:

                final_decision = strongest_decision


    else:
        # there is no dispute; use the choice of the default classifier
        if (default_classifier == 1):

            max_index[0] = max_index[1]
        else:
            if (default_classifier == 2):
                max_index[0] = max_index[2]

    # update confusion matrix data

    if ((label=='0.0') and (final_decision == 0)): 
        NN_ += 1
    if ((label=='1.0') and (final_decision == 1)): 
        SS_ += 1
    if ((label=='2.0') and (final_decision == 2)): 
        RR_ += 1

    if ((label=='0.0') and (final_decision == 1)): 
        NS_ += 1
    if ((label=='0.0') and (final_decision == 2)): 
        NR_ += 1

    if ((label=='1.0') and (final_decision == 0)): 
        SN_ += 1
    if ((label=='1.0') and (final_decision == 2)): 
        SR_ += 1

    if ((label=='2.0') and (final_decision == 0)): 
        RN_ += 1
    if ((label=='2.0') and (final_decision == 1)): 
        RS_ += 1

    # compute performance ------------------------------------------

    if (label=='0.0'):
        if (float(final_decision) == float(label)):
            TP_Nd += 1.0

        if (final_decision != 0):
            FN_Nd += 1.0

    if ((final_decision == 0) and (str(label) != "0.0")):
        FP_Nd += 1.0



    if (label=='1.0'):
        if (float(final_decision) == float(label)):
            TP_Rd += 1.0

        if (final_decision != 1):
            FN_Rd += 1.0

    if ((final_decision == 1) and (str(label) != "1.0")):
        FP_Rd += 1.0



    if (label=='2.0'):
        if (float(final_decision) == float(label)):
            TP_Sd += 1.0

        if (final_decision != 2):
            FN_Sd += 1.0

    if ((final_decision== 2) and (str(label) != "2.0")):
        FP_Sd += 1.0


    line1 = inpt1.readline()

print "Performance of Default Classifier:"
print "TP,FP,FN:"
print TP_N, FP_N, FN_N
print TP_R, FP_R, FN_R
print TP_S, FP_S, FN_S

PN = TP_N / (TP_N + FP_N)
PR = TP_R / (TP_R + FP_R)
PS = TP_S / (TP_S + FP_S)

RN = TP_N / (TP_N + FN_N)
RR = TP_R / (TP_R + FN_R)
RS = TP_S / (TP_S + FN_S)

FN = 2 * PN * RN / (PN + RN)
FR = 2 * PR * RR / (PR + RR)
FS = 2 * PS * RS / (PS + RS)


output_file = input_file + "_result.out"
f = open(output_file,'w')

# calculate the F-score

f.write('\n' + "Default model")
f.write('\n' + "-------------")
f.write('\n' + "P neutral, P racism, P sexism:"+str(PN)+" "+str(PR)+" "+str(PS))
f.write('\n' + "R neutral, R racism, R sexism:"+str(RN)+" "+str(RR)+" "+str(RS))
f.write('\n' + "F neutral, F racism, F sexism:"+str(FN)+" "+str(FR)+" "+str(FS))
f.write('\n')
f.write('\n' + "voting Classifier results:")
f.write('\n' + "    TP,  FP,  FN:")
f.write('\n' + "N:" + str(TP_Nd) + " " + str(FP_Nd) + " " + str(FN_Nd))
f.write('\n' + "R:" + str(TP_Rd) + " " + str(FP_Rd) + " " + str(FN_Rd))
f.write('\n' + "S:" + str(TP_Sd) + " " + str(FP_Sd) + " " + str(FN_Sd))

PN = TP_Nd / (TP_Nd + FP_Nd)
PR = TP_Rd / (TP_Rd + FP_Rd)
PS = TP_Sd / (TP_Sd + FP_Sd)

RN = TP_Nd / (TP_Nd + FN_Nd)
RR = TP_Rd / (TP_Rd + FN_Rd)
RS = TP_Sd / (TP_Sd + FN_Sd)

FNd = 2 * PN * RN / (PN + RN)
FRd = 2 * PR * RR / (PR + RR)
FSd = 2 * PS * RS / (PS + RS)

f.write('\n' + "P neutral, P racism, P sexism:" + str(PN) + " " + str(PR) + " " + str(PS))
f.write('\n' + "R neutral, R racism, R sexism:" + str(RN) + " " + str(RR) + " " + str(RS))
f.write('\n' + "Neutral:" + str(FNd) + " Racism:" + str(FRd) + " Sexism:" + str(FSd))

F = (FN * counterN + FR * counterR+ FS * counterS) / ( counterN + counterR + counterS )
f.write('\n' + "default (unigram) Classifier F:" + str(F) + " F_neutral:" + str(FN) + " F_racism:" + str(FR) + " F_sexism:" + str(FS))

Fd = (FNd * counterN + FRd * counterR+ FSd * counterS) / ( counterN + counterR + counterS )
f.write('\n' + "Combined Classifiers F:" + str(Fd) + " F_neutral:" + str(FNd) + " F_racism:" + str(FRd) + " F_sexism:" + str(FSd))

Pd = (PN * counterN + PR * counterR+ PS * counterS) / ( counterN + counterR + counterS )
f.write('\n' + "Combined Classifiers P:" + str(Pd) + " P_neutral:" + str(PN) + " P_racism:" + str(PR) + " P_sexism:" + str(PS))

Rd = (RN * counterN + RR * counterR+ RS * counterS) / ( counterN + counterR + counterS )
f.write('\n' + "Combined Classifiers R:" + str(Rd) + " R_neutral:" + str(RN) + " R_racism:" + str(RR) + " R_sexism:" + str(RS))

f.write('\n' + "no consensus in cases:" + str(noconsensus) + ",  re-labelings:" + str(re_labels) + ", succ. re-labelings:" + str(ok_re_labels))

f.write('\n' + "NN,RR,SS,NS,NR,RS,RN,SN,SR" + str(NN_) + " " + str(RR_) + " " + str(SS_) + " " + str(NS_) + " " + str(NR_) + " " + str(RS_) + " " + str(RN_) + " " + str(SN_) + " " + str(SR_))

f.close()
