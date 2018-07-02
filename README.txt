Useful instructions for re-producing the experiment documented in the report: "Detecting Offensive Lnguage in tweets ..."


s/w requirmenets:

a) keras, version 1.2.2
b) tensorflow, version 
c) moses decoder can be downloaded from: http://www.statmt.org/moses/?n=Development.GetStarted


nessesary source files:
classifier.py
run_loop_NR.sh
run_loop_NRS.sh
run_loop_NS.sh
run_loop_O.sh
run_loop_RS.sh


The process of running the expertiment step-by-step:

1) Install moses decoder toolkit that can be found at: "http://www.statmt.org/moses/?n=Development.GetStarted"
 and modify the tokenizer_cmd line accordinly in tok_lib.py so that it points to the proper location where the tokenizer.perl script has been installed.

2) The hate-speech dataset to use for reproducting the experiment can be taken from the address indicated in the following publication:

Zeerak Waseem and Dirk Hovy, Hateful symbols or hateful people? predictive features for hate speech detection on twitter. In Proceedings of the NAACL Student Research Workshop, San Diego, California, June 2016. Association for Computational Linguistics.

The input data files must be in the proper csv format, with each line structured like: tweeter_ID,label,textual_content e.g. 5878947239423749,none,"Hello world"

Please note that the data for every class must be provided in 3 separate files. The value of the label should be either "none", "sexism" or "racism".
The filenames to use for those files are:
"tweets_hate_speech_none.csv",
"tweets_hate_speech_sexism.csv",
"tweets_hate_speech_racism.csv".


Please note that the number of records on each file must be declared into the source code file (classifier.py), by setting the appropriate values in the following 3 valiables:
load_none_train   = <number of records in _none file>
load_racism_train = <number of records in _racism file>
load_sexism_train = <number of records in _sexism file>

If you wish use differenet filenames, then the value of each "input_file" variable in ht_lib.py must be modified accordingly.

Trying the code on a dataset with different size, will requite doing the proper modifications to these lines in accordinly:

load_none_train    =  <number of neutral profiles>
load_racism_train  =  <number of racism  profiles>
load_sexism_train  =  <number of sexism  profiles>

3) The tweet ids used in the dataset, along with the ids of the users posted the tweets must also be included in a separate file called 'hate_speech_tweets_users.csv'. The file must have a structure of 2 column csv, such as: <tweetID>,<userID>

The user_ids can be retrieved by crawling the hate-speech dataset in twitter, with the provision of the tweet_ids. You can use your own crawler for retrieving such information from twitter.

4) The various user features that are included into the training data must be provided in a separated file with name: user_class_ratio.csv. The feature information must be provided in a csv file with the following format:
<user_id>,<label>,<ratio value>
The acceptable values for the <label> field are: none, racism or sexism.

The contents of the features file can be produced by running the following helper scripts:
(software dependencies:unix shell and "q text" toolkit)
In the unix prompt type:

> cat tweets_hate_speech_none.csv tweets_hate_speech_sexism.csv tweets_hate_speech_racism.csv > tweets_hate_speech.csv

> cat tweets_hate_speech.csv | awk -F',' '{ print $1","$2 }' > labels.csv

> q -d, "select u.c2,count(*) from hate_speech_tweets_users.csv as u group by u.c2" > user_count.csv

> q -d, "select u.c2,l.c2,count(*) from hate_speech_tweets_users.csv as u join labels.csv as l where u.c1=l.c1 group by u.c2,l.c2" > label_count.csv

> q -d, "select c.c1,f.c2,(1.0 * f.c3/c.c2) from label_count.csv as f join user_count.csv as c where c.c1=f.c1" > user_class_ratio.csv


5) Perform the classification by running the following 5 scripts.

run_loop_NR.sh,  this is for running the experiment with the inclusion of 'Neutral' and 'Sexism' user-based features alone.
run_loop_NRS.sh, this is for running the experiment with the inclusion of all user-based features 'Neutral' and 'Sexism' and 'Racism'.
run_loop_NS.sh,  this is for running the experiment with the inclusion of 'Neutral' and 'Sexism' user-based features alone.
run_loop_O.sh,   this is for running the experiment with no  inclusion of any user-based features.
run_loop_RS.sh,  this is for running the experiment with the inclusion of 'Racism' and 'Sexism' user-based features alone.

In the above scripts every experiment is set to run for a total number of 15 times (it can be easily changed by modifying the script)


6) Compute the performance of every single classifier by running the following script in the unix prompt: "run_results.sh". 

The produced report files contain the computed F-score values for each classifier:
O.csv_result.out    for no inclusion of additional features 
NS.csv_result.out   for Neutral and Sexism feature included
RS.csv_result.out   for Racism and Sexism feature included
NR.csv_result.out   for Neutral and Racism feature included
NRS.csv_result.out  for Neutral,Sexism,Racism features included

The performance figure reported in the published document is marked as: 'Combined F:' in the above files.

7) Compute the performance of the 3-input ensembles by running the following script in the unix prompt: "run_results_3_ensembles.sh". 
This script combines together the output produced by the scripts in step 5. There should be a number of output files equal to the 3-classifier ensembles included in the published document.
The names of the output files have the form: mixfile_<classifier1>_<classifier2>_<classifier3>csv_result.out denoting which classifiers used as input.
e.g.:
mixfile_O_NSR_NS.csv_result.out is referred to the result of the ensemble, composed of the following 3 inputs: 

i)   No-user-features
ii)  Neutral+Sexism+Racism features
iii) Neutral+Sexism features

The contents of the output files include the performance figure for the enssemble, expressed in various metrics, including the F-score.
The values reported in the published document are marked as: 'Combined classifiers F:' in the above files.


8) Compute the performance of the 5-input ensemble by running the script:
run_results_5_ensemble.sh

This script combines together the output produced by the scripts in step 5 and it prints the output on the screen.
The output incluces the F-score value reported in the published document.


