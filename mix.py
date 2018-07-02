import csv
import sys
import numpy as np

lines1 = []
lines2 = []
lines3 = []

lines1a = []
lines2a = []
lines3a = []

cmd_args = len(sys.argv)

if (cmd_args<5):
   print "Not enough arguments."
   sys.exit(0) 

list1 = sys.argv[1]
list2 = sys.argv[2]
list3 = sys.argv[3]
output = sys.argv[4]


# list 1
f = open(list1)
lines1 = f.readlines()
f.close()

# list 2
f = open(list2)
lines2 = f.readlines()
f.close()

# list 3
f = open(list3)
lines3 = f.readlines()
f.close()


for i in lines1:
    i = i.rstrip("\n")
    lines1a.append(i)

for i in lines2:
    i = i.rstrip("\n")
    lines2a.append(i)

for i in lines3:
    i = i.rstrip("\n")
    lines3a.append(i)

print lines1a
print lines2a
print lines3a

# combine lists
counter = 0
for j in lines1a:
    for k in lines2a:
        for l in lines3a:

            counter += 1
            print j,",",k,",",l

            # load the contents of file j
            print j
            with open(j, 'rb') as f:
                reader = csv.reader(f)
                your_list1 = list(reader)

            # load the contents of file k
            with open(k, 'rb') as f:
                reader = csv.reader(f)
                your_list2 = list(reader)

            # load the contents of file l
            with open(l, 'rb') as f:
                reader = csv.reader(f)
                your_list3 = list(reader)

            columns = np.hstack((your_list1, your_list2))
            columns = np.hstack((columns, your_list3))

            f_handle = file(output, 'a')
            np.savetxt(f_handle, (columns),delimiter=',',fmt="%s")
            f_handle.close()


print counter

print "Check output file:", output
