# policy using 3 classifiers

from statistics import mode, StatisticsError
import numpy as np
import sys

import random
input_file1 = "all_unigrams_NR_relu.csv"
input_file2 = "all_unigrams_NRS_relu.csv"
input_file3 = "all_unigrams_RS_relu.csv"
input_file4 = "all_unigrams_NS_relu.csv"
input_file5 = "all_unigrams_relu.csv"


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

lineCounter = 0

counterN = 0
counterR = 0
counterS = 0

noconsensus = 0
re_labels = 0
ok_re_labels = 0

default_classifier = 0


inpt1 = open(input_file1, "r+")
inpt2 = open(input_file2, "r+")
inpt3 = open(input_file3, "r+")
inpt4 = open(input_file4, "r+")
inpt5 = open(input_file5, "r+")

line1 = inpt1.readline()
line2 = inpt2.readline()
line3 = inpt3.readline()
line4 = inpt4.readline()
line5 = inpt5.readline()

while (line1 != ''):

    if (line1 == '\n'):

        print "run out of records."
        sys.exit(0)

    fields1 = line1.split(",")
    fields2 = line2.split(",")
    fields3 = line3.split(",")
    fields4 = line4.split(",")
    fields5 = line5.split(",")

    lineCounter += 1

    tid = fields1[0]
    if (not (tid == fields2[0] == fields3[0])):
       print "Oooops. Inconsistent input files at line:", lineCounter
       break
       sys.exit(0)

    if (not (fields1[1] == fields2[1] == fields3[1])):
       print "Oooops. Inconsistent labels in input files at line:", lineCounter
       sys.exit(0)

    neutral_1 = float(fields1[2])
    neutral_2 = float(fields2[2])
    neutral_3 = float(fields3[2])
    neutral_4 = float(fields4[2])
    neutral_5 = float(fields5[2])

    racism_1 = float(fields1[3])
    racism_2 = float(fields2[3])
    racism_3 = float(fields3[3])
    racism_4 = float(fields4[3])
    racism_5 = float(fields5[3])

    sexism_1 = float(fields1[4])
    sexism_2 = float(fields2[4])
    sexism_3 = float(fields3[4])
    sexism_4 = float(fields4[4])
    sexism_5 = float(fields5[4])

    label = fields1[1]
    # compute f-score here and compare the result with 1st classifier alone

    lst_1 = [ neutral_1, racism_1, sexism_1 ]
    lst_2 = [ neutral_2, racism_2, sexism_2 ]
    lst_3 = [ neutral_3, racism_3, sexism_3 ]
    lst_4 = [ neutral_4, racism_4, sexism_4 ]
    lst_5 = [ neutral_5, racism_5, sexism_5 ]

    max_index = np.empty([5], dtype=int)


    max_value = max(lst_1)
    max_index[0] = lst_1.index(max_value)

    max_value = max(lst_2)
    max_index[1] = lst_2.index(max_value)

    max_value = max(lst_3)
    max_index[2] = lst_3.index(max_value)

    max_value = max(lst_4)
    max_index[3] = lst_4.index(max_value)

    max_value = max(lst_5)
    max_index[4] = lst_5.index(max_value)

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

    if not (max_index[0] == max_index[1] == max_index[2] == max_index[3] == max_index[4]): # 99

            noconsensus += 1

            # averaging

            s_neutral = neutral_1 + neutral_2 + neutral_3 + neutral_4 + neutral_5
            s_racism  = racism_1  + racism_2  + racism_3 + racism_4 + racism_5
            s_sexism  = sexism_1  + sexism_2  + sexism_3 + sexism_4 + sexism_5

            values = [s_neutral, s_racism, s_sexism]

            # the index of the max value
            average_decision = np.argmax(values)

            # choose the most confident classifier
            m1 = max(lst_1)
            m2 = max(lst_2)
            m3 = max(lst_3)
            m4 = max(lst_4)
            m5 = max(lst_5)

            most_conf_classifier = np.argmax([m1, m2, m3, m4, m5])

            strongest_decision = max_index[most_conf_classifier]

            try:
                voting_decision = mode([max_index[0], max_index[1], max_index[2], max_index[3] , max_index[4]])

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
                        print "LOST ANYWAY"

            except StatisticsError:
                print "hard decision. :-(. choose the most confident one."


                final_decision = strongest_decision
                print final_decision

    else:
        # there is no dispute; use the choice of the default classifier
        if (default_classifier == 1):

            max_index[0] = max_index[1]
        else:
            if (default_classifier == 2):
                max_index[0] = max_index[2]


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
    line2 = inpt2.readline()
    line3 = inpt3.readline()
    line4 = inpt4.readline()
    line5 = inpt5.readline()

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

# re-calculate F score here

print "Default model"
print "--------------"
print "P neutral, P racism, P sexism:", PN, PR, PS
print "R neutral, R racism, R sexism:", RN, RR, RS
print "F Neutral:", FN, " F Racism:", FR, " F Sexism:", FS

print
print "voting Classifier results:"
print "TP,FP,FN:"
print TP_Nd, FP_Nd, FN_Nd
print TP_Rd, FP_Rd, FN_Rd
print TP_Sd, FP_Sd, FN_Sd

PN = TP_Nd / (TP_Nd + FP_Nd)
PR = TP_Rd / (TP_Rd + FP_Rd)
PS = TP_Sd / (TP_Sd + FP_Sd)

RN = TP_Nd / (TP_Nd + FN_Nd)
RR = TP_Rd / (TP_Rd + FN_Rd)
RS = TP_Sd / (TP_Sd + FN_Sd)

FNd = 2 * PN * RN / (PN + RN)
FRd = 2 * PR * RR / (PR + RR)
FSd = 2 * PS * RS / (PS + RS)

print "P neutral,P racism, P sexism:", PN, PR, PS
print "R neutral,R rasicm, R sexism:", RN, RR, RS

print "Neutral:", FNd, " Racism:", FRd, " Sexism:", FSd


F = (FN * counterN + FR * counterR+ FS * counterS) / ( counterN + counterR + counterS )
print "default (unigram) Classifier F:", F, " F_neutral:", FN, " F_racism:", FR, " F_sexism:", FS

Fd = (FNd * counterN + FRd * counterR+ FSd * counterS) / ( counterN + counterR + counterS )
print "Combined Classifiers F:", Fd, " F_neutral:", FNd, " F_racism:", FRd, " F_sexism:", FSd

Pd = (PN * counterN + PR * counterR+ PS * counterS) / ( counterN + counterR + counterS )
print "Combined Classifiers P:", Pd, " P_neutral:", PN, " P_racism:", PR, " P_sexism:", PS

Rd = (RN * counterN + RR * counterR+ RS * counterS) / ( counterN + counterR + counterS )
print "Combined Classifiers R:", Rd, " R_neutral:", RN, " R_racism:", RR, " R_sexism:", RS

print "noconsensus in cases:", noconsensus, ",  re-labelings:", re_labels, ", succ. re-labelings:", ok_re_labels


