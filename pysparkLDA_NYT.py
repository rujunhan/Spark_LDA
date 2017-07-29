#### run this line in terminal: spark-submit --packages com.databricks:spark-xml_0.10:0.4.1 testLDA.py !!! 

from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SQLContext
import copy
import numpy as np
from operator import add
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import ngrams
import datetime
import subprocess

### import spark context
App_Name = "LDA.PY"
conf = SparkConf().setAppName(App_Name)
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)


### Define global variables
NGRAM = 5
data_year = '2007'
filter_path = 'filter_files'
topicMatrix_file = 'topics_0729.npy'
train_flag = 0   ## set executor-memory to 10G
topic_n = 10
# tech sector key words
keywords = ['NOVELL', 'H.P.', 'QLOGIC', 'JABIL CIRCUIT', 'INTEL', 'SYMANTEC', 'AUTODESK', 'XEROX', 'CORNING',
'COMPUTER SCIENCES', 'CA INC', 'YAHOO', 'INTL BUSINESS MACHINES', 'MICRON TECHNOLOGY','QUALCOMM', 'TELLABS', 'NATIONAL SEMICONDUCTOR', 'COMPUWARE',
'CIENA', 'XILINX', 'ANALOG DEVICES', 'DELL', 'INTUIT', 'LINEAR TECHNOLOGY', 'TERADYNE', 'B.M.C.', 'FIDELITY NATIONAL INFO SVCS', 'MICROSOFT', 'WESTERN UNION',
'NOVELLUS', 'SUN MICROSYSTEMS', 'LSI', 'ADOBE', 'APPLE INC', 'GOOGLE', 'EMC', 'APPLIED MATERIALS', 'KLA-TENCOR', 'TEXAS INSTRUMENTS', 'ELECTRONIC DATA', 'ALTERA',
'ADVANCED MICRO DEVICES', 'COGNIZANT', 'MOLEX', 'NETAPP', 'AUTOMATIC DATA PROCESSING', 'PAYCHEX', 'SANDISK', 'LEXMARK', 'AFFILIATED', 'NVIDIA', 'BROADCOM', 'MOTOROLA',
'CONVERGYS', 'FISERV', 'VIAVI', 'IAC/INTERACTIVECORP', 'JUNIPER', 'ELECTRONIC ARTS', 'CISCO', 'MONSTER WORLDWIDE', 'EBAY', 'KODAK', 'UNISYS', 'ORACLE','CITRIX SYSTEMS', 'VERISIGN']


def parse_xml(x):

	try:
		return x[0][-1][1][0]
        except:
		return ""

def process_corpus(x,N):
        # remove punctuations and numbers
	text = re.sub("[^a-zA-Z]", " ", x)

        # tokenize corpus
        word_list = word_tokenize(text)
        
	# filter out stop words
        filtered_words = [str(w) for w in word_list if w.lower() not in stopwords.words('english')]
        
	# distinguish upper and lower case
	cap_words = ['zzz_'+ w.lower() for w in filtered_words if not w.islower()]
	lower_words = [w for w in filtered_words if w.islower()]
	filtered_words = [w.lower() for w in filtered_words]
        
	result = cap_words
	result.extend(lower_words)
        # create n-grams
	for x in range(2, N+1):
		result.extend(['zzz_' + str('_'.join(ngr)) for ngr in ngrams(filtered_words, x)])

	return result
	
def get_corpus(filepath):
        # return a rdd of all text corpus with ngrams
        textfiles = sqlContext.read.format("com.databricks.spark.xml").option("rowTag", "body.content").load(filepath + "*.xml")
        return textfiles.map(lambda x: parse_xml(x))

def create_vocabulary(path):
	vcb = get_corpus(path).map(lambda x: process_corpus(x,NGRAM)).flatMap(lambda x: x).distinct().collect()
	return sorted(vcb)

def word_count(text, vcb):        
	return text.filter(lambda x: x in vcb).map(lambda x: (vcb.index(x)+1, 1)).reduceByKey(add).collect()
	

def bag_of_ngrams(data, vcb): #should be path instead of x if trying to create vocabulary
	#all_text = get_corpus(path).collect()
	BoN = []
	if not data:
		return []
        else:
		for tx in data:
			ngrams = process_corpus(tx, NGRAM)
			BoN.append(word_count(sc.parallelize(ngrams, len(ngrams)), vcb))
	return BoN
        
def runLDA(filepath, n):
	data = sc.textFile(filepath)
        n_vcb = int(data.take(2)[1])

        parsedData = data.map(lambda line: line.strip().split(' ')).filter(lambda x: len(x) > 2).map(lambda x: (int(x[0])-1, (int(x[1])-1, float(x[2])))).groupByKey().mapValues(list)
	corpus = parsedData.map(lambda x: [x[0], Vectors.sparse(n_vcb, x[1])]).cache()

	ldaModel = LDA.train(corpus, k=n)

	print "Learned topics (as distributions over vocab of " + str(ldaModel.vocabSize()) + " words):"
	print ldaModel.describeTopics(maxTermsPerTopic=20)
        return ldaModel.topicsMatrix()

def normalize(arr):
	return [x/np.sum(arr) for x in arr]

def docTopics(docs, topicMatrix):
	# for each topic
	# sum probablity of words in corpus
        # normalize so that probility of topics sum to 1
	
	n_vcb = topicMatrix.shape[0]
	return [normalize(Vectors.sparse(n_vcb, x).dot(topicMatrix)) for x in docs]
	

def inSector(x, keywords):
	# check if corpus has keywords
	text = re.sub("[^a-zA-Z]", " ", x)
			
	return bool(set([w.lower() for w in str(text).split(' ')]) & set([kw.lower() for kw in keywords]))
	
def select_articles(path, keywords):
	# check if filtered files exist already
        cmd=['hdfs', 'dfs', '-ls', path]
	if filter_path in subprocess.check_output(cmd).strip():
		return
	else:
		textfiles = sqlContext.read.format("com.databricks.spark.xml").option("rowTag", "body.content").load(path + "*.xml")
		data = textfiles.map(lambda x: parse_xml(x)).filter(lambda x: inSector(x, keywords)).saveAsTextFile(path + filter_path)
       		return

def parse_dates(data_year):

        cmd = ['hdfs', 'dfs', '-ls']
	cmd_y = cmd+[data_year+'/']

	months = [m[-2:] for m in subprocess.check_output(cmd_y).strip().split('\n')][2:]
        
	pathlist = []
	for m in months:
		cmd_m = cmd + [data_year+'/'+m+'/']

		days = [d[-10:] for d in subprocess.check_output(cmd_m).strip().split('\n')][1:]

                for d in days:
                        pathlist.append(d+'/')
	return pathlist

def main():
	
	### if re-train, load pre-processed text corpus and train LDA
	if train_flag:
		corpuspath = "lda/docword.nyt.txt" 
		topicMatrix = runLDA(corpuspath, topic_n)
		np.save('topics_0729.npy', topicMatrix)
	else:
		topicMatrix = np.load(topicMatrix_file)
	
	# load or create vocabulary
	vocabulary = sc.textFile('lda/nyt_voc.txt').map(lambda x: str(x)).collect()

	### parse through directory to obtain dates
	path_list = parse_dates(data_year)
	
	### create topic dictionary: key by date
	topic_dictionary = {}
	for filepath in path_list:
		# filter article with sector key words
		
		select_articles(filepath, keywords)
		data = sc.textFile(filepath + filter_path).collect()
	
		# create bag of n-grams for each corpus 
		BoN = bag_of_ngrams(data, vocabulary)

		#compute corpus topics with results (topicMatrix) from trained LDA
		topic_dictionary[filepath] = docTopics(BoN, topicMatrix)
	
	timestamp = str(datetime.datetime.now().month) + str(datetime.datetime.now().day) 
	np.save('result_'+timestamp+'.npy', topic_dictionary)
                                                                                                            
if __name__ == "__main__":
	main()


