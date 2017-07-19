#### run this line in terminal: spark-submit --packages com.databricks:spark-xml_2.10:0.4.1 testLDA.py !!! 

from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SQLContext

import numpy as np
import nltk
import string
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import ngrams

import os
import subprocess

# import spark context
App_Name = "LDA.PY"
conf = SparkConf().setAppName(App_Name)
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

#obtain all files in a diretory
def get_filenames(path):
	cmd = ['hdfs', 'dfs', '-ls', path]
	files = [f.split(' ')[-1] for f in subprocess.check_output(cmd).strip().split('\n')]
        return files[1:]

#hedline = sqlContext.read.format("com.databricks.spark.xml").option("rowTag", "hedline").load("lda/temp/1815836.xml")

#print hedline.take(1)[0][0]

# get xml content
def parse_xml(filepath):
	print filepath
	content = sqlContext.read.format("com.databricks.spark.xml").option("rowTag", "body.content").load(filepath)
	try:
		result = content.take(1)[0][0][-1][1]
        # no full text !!!
	except:
		result = " "
        return result

def process_corpus(x,N):
        # remove punctuations and numbers
	text = re.sub("[^a-zA-Z]", " ", x)
        # tokenize corpus
        word_list = word_tokenize(text)
        # filter out stop words
        filtered_words = [w for w in word_list if w.lower() not in stopwords.words('english')]
        # distinguish cap words
        filtered_words = [w if w.islower() else 'zzz_' + w.lower() for w in word_list]
        result = filtered_words
        # create n-grams
        n_list = sc.parallelize(range(2, N+1), N-1)
        n_grams = n_list.map(lambda x: ['zzz'+'_'.join(list(ngr)) for ngr in ngrams(filtered_words, x)]).flatMap(lambda x: x).collect()	
        for ng in n_grams:
		result.append(ng)
	return result

def create_vocabulary(path):
	files = get_filenames(path)
	vcb = []
        #sc.broadcast(vcb)
	for f in files:
		#print files
		#print "\n"
	#par = sc.parallelize(range(0, len(files)))
	#corpus = par.map(lambda x: parse_xml(files[x])).collect()	
		corpus = parse_xml(f)
		corpus = np.unique(process_corpus(corpus,5))
	        #all_corpus.append(corpus)
		vcb.append(corpus)
	return sorted(np.unique(vcb))

def bag_of_ngrams(text):
	result = text
	return result
	
vocabulary = create_vocabulary('lda/temp/')
print(vocabulary)

#text = process_corpus((" ").join(content.take(1)[0][0][-1][1]),5)
#print text
#print len(text)
'''
# Load and parse the data
data = sc.textFile("lda/nips.txt")
parsedData = data.map(lambda line: Vectors.dense([float(x) for x in line.strip().split(' ')]))
# Index documents with unique IDs
corpus = parsedData.zipWithIndex().map(lambda x: [x[1], x[0]]).cache()

# Cluster the documents into three topics using LDA
ldaModel = LDA.train(corpus, k=3)

# Output topics. Each is a distribution over words (matching word count vectors)
print("Learned topics (as distributions over vocab of " + str(ldaModel.vocabSize()) + " words):")
topics = ldaModel.topicsMatrix()
for topic in range(3):
    print("Topic " + str(topic) + ":")
    for word in range(0, ldaModel.vocabSize()):
        print(" " + str(topics[word][topic]))

# Save and load model
model.save(sc, "myModelPath")
sameModel = LDAModel.load(sc, "myModelPath")
'''
