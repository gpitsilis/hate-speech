Useful information for re-producing the experiment documented in the reports cited as:

Pitsilis, G.K, Ramampiaro, H., Langseth, H., “Detecting Offensive Language in Tweets Using Deep Learning”, 2018/1/13, arXiv preprint arXiv:1801.04433. https://arxiv.org/pdf/1801.04433

Pitsilis, G.K., Ramampiaro, H., & Langseth, H. (2018),"Effective hate-speech detection in Twitter data using recurrent neural networks", Applied Intelligence, 48(12), 4730-4742.

Software requirements:
----------------------

a) keras (version 1.2.2)
b) tensorflow
c) python 2.7x
d) moses decoder ( only the part related to tokenization )
e) q text toolkit (provisionally) from: http://harelba.github.io/q/
f) access to linux shell

Published source code files:
----------------------------

classifier.py
tok_lib.py
ht_lib.py
validate.py
mix.py
validate_mix.py
merging_5.py

sort.sh
run_loop_NR.sh
run_loop_NRS.sh
run_loop_NS.sh
run_loop_O.sh
run_loop_RS.sh
run_results.sh
run_results_3_ensemble.sh
run_results_5_ensemble.sh

Running the experiment step-by-step:
------------------------------------

1) Install moses decoder toolkit that can be found at: http://www.statmt.org/moses/?n=Development.GetStarted
Once moses is installed, you will need to replace the full path of the tokenizer.perl file in tok_lib.py with the one pointing to the location 
where the  script is installed in your hard disk.

2) Acquire the hate-speech dataset. It can be taken from the link advertised in the following publication:

Zeerak Waseem and Dirk Hovy, Hateful symbols or hateful people? predictive features for hate speech detection on twitter. In Proceedings of the NAACL 
Student Research Workshop, San Diego, California, June 2016. Association for Computational Linguistics.

Note that the input data files must be provided in the proper csv format, with each line structured like: <tweeter_ID>,<label>,<textual_content>
The <label> should be either "none", "sexism" or "racism".
e.g.: 5878947239423749,none,"Hello world"

Also, the input data aquired from the above file must be converted and provided in 3 separate files, one for each class, with the following names:
"tweets_hate_speech_none.csv",
"tweets_hate_speech_sexism.csv",
"tweets_hate_speech_racism.csv".

Also note that the number of records to be read from each class file is hardcoded in the source code file (classifier.py). As such, the proper 
values in the following 3 variables must be set:
load_none_train   = <number of records to read from the neutral tweets>
load_racism_train = <number of records to read from racism tweets>
load_sexism_train = <number of records to read from sexism tweets>

If you wish to use different filenames for the above input data files, then you will need to modify the content of 
the 'input_file' variable in 'ht_lib.py' accordingly.

If you wish to try the code on the dataset with different number of records for racism/sexism/neutral, then you will need to modify these lines 
accordinlgy, giving the proper values:

load_none_train    =  <number of neutral profiles>
load_racism_train  =  <number of racism  profiles>
load_sexism_train  =  <number of sexism  profiles>

3) Provide the user_ids.
The tweet ids used in the dataset, along with the ids of the users which have posted those tweets must also be included into a separate file with name: 
'hate_speech_tweets_users.csv'. Every line in that file must be structured like: <tweetID>,<userID>

The userIDs can be retrieved by crawling the hate-speech dataset in twitter, given the tweet_ids. You may use your own software solution for crawling on 
twitter and retrieve the above information.

4) Provide the user-related features.
The various user features to be included into the training data must be provided into a separate file with name: user_class_ratio.csv in the following format:
<user_id>,<label>,<ratio value>
The acceptable values for the <label> field are: “none”, “racism” or “sexism”.

The contents of the above file can be generated out from the existing data files. The following helper scripts running in the linux shell prompt can 
used for that purpose ( requires the installation of the q text toolkit ):


$ cat tweets_hate_speech_none.csv tweets_hate_speech_sexism.csv tweets_hate_speech_racism.csv > tweets_hate_speech.csv

$ cat tweets_hate_speech.csv | awk -F',' '{ print $1","$2 }' > labels.csv

$ q -d, "select u.c2,count(*) from hate_speech_tweets_users.csv as u group by u.c2" > user_count.csv

$ q -d, "select u.c2,l.c2,count(*) from hate_speech_tweets_users.csv as u join labels.csv as l where u.c1=l.c1 group by u.c2,l.c2" > label_count.csv

$ q -d, "select c.c1,f.c2,(1.0 * f.c3/c.c2) from label_count.csv as f join user_count.csv as c where c.c1=f.c1" > user_class_ratio.csv

5) Perform the classification.

Run the following 5 shell scripts once:

run_loop_NR.sh  : This is for running the experiment with the inclusion of 'Neutral' and 'Sexism' user-based features alone.
run_loop_NRS.sh : This is for running the experiment with the inclusion of all user-based features 'Neutral','Sexism' and 'Racism'.
run_loop_NS.sh  : This is for running the experiment with the inclusion of 'Neutral' and 'Sexism' user-based features alone.
run_loop_O.sh   : This is for running the experiment with no  inclusion of any user-based features.
run_loop_RS.sh  : This is for running the experiment with the inclusion of 'Racism' and 'Sexism' user-based features alone.

In the above scripts every experiment is set to run for 15 times in total. (The number of times can be changed by doing the proper alterations in the script)


6) Compute the performance of every single classifier by running the following script in the linux prompt:
$ run_results.sh

The performance for each classifier is recorded into the following files and it includes the computed F-score (marked as: 'Combined F:'):

O.csv_result.out      for the case of no-inclusion of additional features
NS.csv_result.out    for Neutral and Sexism features included
RS.csv_result.out    for Racism and Sexism features included
NR.csv_result.out    for Neutral and Racism features included
NRS.csv_result.out  for Neutral,Sexism,Racism features included

7) Compute the performance of the 3-input ensembles by running the following script in the linux prompt:
$ run_results_3_ensemble.sh
This script combines together the output produced by the scripts in step 5.

The performance figure of every ensemble is recorded into a file with filename denoting the classifiers used as input to that ensemble. 
The filename has the form: mixfile_<classifier1>_<classifier2>_<classifier3>csv_result.out.
e.g.: mixfile_O_NSR_NS.csv_result.out is referred to the result of the ensemble the has the following 3 inputs:

i)   No-user-features
ii)  Neutra,Sexism and Racism features
iii) Neutral and Sexism features

In total, 10 output files will be produced ( as many as the ensembles in the published document ) containing the various metrics associated 
with the performance, such as the F-score. The later is marked as: 'Combined classifiers F:' in the above files.


8) Compute the performance of the 5-input ensemble by running the script:
run_results_5_ensemble.sh

This script combines together the output produced by the scripts in step 5. The performance figure of the ensemble (includes the F-score) 
is send to the standard output ( standard console ).

Contact information: georgios.pitsilis@gmail.com

If you use this work for academic research with published results, we will be grateful if you cite the following paper:

Pitsilis, G.K., Ramampiaro, H., & Langseth, H. (2018),"Effective hate-speech detection in Twitter data using recurrent neural networks", 
Applied Intelligence, 48(12), 4730-4742.

in bibtext format:

@article{10.1007/s10489-018-1242-y,
author = {Pitsilis, Georgios K. and Ramampiaro, Heri and Langseth, Helge},
title = {Effective Hate-Speech Detection in Twitter Data Using Recurrent Neural Networks},
year = {2018},
issue_date = {December 2018},
publisher = {Kluwer Academic Publishers},
address = {USA},
volume = {48},
number = {12},
issn = {0924-669X},
url = {https://doi.org/10.1007/s10489-018-1242-y},
doi = {10.1007/s10489-018-1242-y},
journal = {Applied Intelligence},
month = dec,
pages = {4730–4742},
numpages = {13},
keywords = {Text classification, Recurrent neural networks, Micro-blogging, Twitter, Hate-speech, Deep learning}
}

  


