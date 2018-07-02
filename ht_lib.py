
def load_neutral(load_none_train, corpus,ids, uID_dict, uIDs):

    print "Loading neutral profiles (training set) ....."

    input_file = 'hate_speech_none.csv'

    # read values and save after rescaling
    leg_inpt = open(input_file,"r+")

    line = leg_inpt.readline()

    counter = load_none_train

    while (line != ''):
        if (line == '\n'):
            print "Ran out of neutral profiles !"
            sys.exit(1)

        fields   = line.split(",none,")
        id       = fields[0]
        sentence = fields[1]

        sentence = sentence[1:-1]
        sentence = sentence.strip('"')

        corpus.append(sentence)
        ids.append(id)

        try:
          uid = uID_dict[id]
          uIDs.append(uid)
        except KeyError:
          uIDs.append(0)

        line = leg_inpt.readline()

        counter -= 1
        if (counter <= 0):
            break

    leg_inpt.close()

    return

# ----------------------------------------------------------------

def load_racism(load_racism_train, corpus,ids, uID_dict, uIDs):

    print "Loading racism profiles (training set) ....."

    input_file = 'hate_speech_racism.csv'

    # read values and save after rescaling
    race_inpt = open(input_file,"r+")

    line = race_inpt.readline()

    counter = load_racism_train

    while (line != ''):

        if (line == '\n'):
            print "Ran out of racism profiles for training !"
            sys.exit(1)

        fields   = line.split(",racism,")
        tags = fields[1]

        id       = fields[0]
        sentence = fields[1]

        # remove leading and trailing quotes
        sentence = sentence[1:-1]
        sentence = sentence.strip('"')

        corpus.append(sentence)
        ids.append(id)

        try:
          uid = uID_dict[id]
          uIDs.append(uid)
        except KeyError:
          uIDs.append(0)

        line = race_inpt.readline()
        counter -= 1

        if (counter <= 0):
            break

    race_inpt.close()

    return

# ----------------------------------------------------------------

def load_sexism(load_sexism_train, corpus, ids, uID_dict, uIDs):

    print "Loading sexism profiles (training set) ....."

    input_file = 'hate_speech_sexism.csv'

    # read values and save after rescaling
    sex_inpt = open(input_file,"r+")

    line = sex_inpt.readline()

    counter = load_sexism_train

    while (line != ''):

        if (line == '\n'):
            print "Ran out of sexism profiles for training !"
            sys.exit(1)

        fields   = line.split(",sexism,")
        tags     = fields[1]

        id       = fields[0]
        sentence = fields[1]

        sentence = sentence[1:-1]
        sentence = sentence.strip('"')

        corpus.append(sentence)
        ids.append(id)

        try:
          uid = uID_dict[id]
          uIDs.append(uid)
        except KeyError:
          uIDs.append(0)

        line = sex_inpt.readline()
        counter -= 1

        if (counter <= 0):
            break

    sex_inpt.close()

    return
