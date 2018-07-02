def init_tokenizer():

    # replace /home/username/deep-learning/tokenizer/tokenizer.perl with the proper location of the rokenizer.perl script
    tokenizer_cmd = ['/usr/bin/perl','/home/username/deep-learning/tokenizer/tokenizer.perl', '-l', 'en', '-q', '-']

    return tokenizer_cmd
