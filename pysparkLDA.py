#### run this line in terminal: spark-submit --packages com.databricks:spark-xml_0.10:0.4.1 testLDA.py !!! 

from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SQLContext
import copy
import numpy as np
from operator import add
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import ngrams
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

def parse_xml(x):

	try:
		return x[0][-1][1][0]
        except:
		return ""

def process_corpus(x,N):
        # remove punctuations and numbers
	text = re.sub("[^a-zA-Z]", " ", x)
        # tokenize corpus
        # word_list = word_tokenize(text)
        filtered_words = text.split(" ")
        # filter out stop words
        #filtered_words = [w for w in word_list if w.lower() not in stopwords.words('english')]
        # distinguish cap words
        #filtered_words = [w if w.islower() else 'zzz_' + w.lower() for w in word_list]
        result = copy.copy(filtered_words)
        # create n-grams
	for x in range(2, N+1):
		result.extend(['zzz' + str('_'.join(ngr)) for ngr in ngrams(filtered_words, x)])

	return result
	
def get_corpus(filepath):
        # return a rdd of all text corpus with ngrams
        textfiles = sqlContext.read.format("com.databricks.spark.xml").option("rowTag", "body.content").load(filepath + "*.xml")
        return textfiles.map(lambda x: parse_xml(x)).map(lambda x: process_corpus(x,5))

def create_vocabulary(path):
	vcb = get_corpus(path).flatMap(lambda x: x).distinct().collect()
	return sorted(vcb)

def word_count(text, vcb):        
	return text.filter(lambda x: x in vcb).map(lambda x: (vcb.index(x)+1, 1)).reduceByKey(add).collect()
	

def bag_of_ngrams(path, vcb):
	all_text = get_corpus(path).collect()
	for tx in all_text[0:2]:
		print word_count(sc.parallelize(tx, len(tx)), vcb)

def runLDA(filepath, n):
	data = sc.textFile(filepath)
        n_vcb = data.take(2)[1]
	print data.take(2)[0]
        parsedData = data.map(lambda line: line.strip().split(' ')).filter(lambda x: len(x) > 2).map(lambda x: (int(x[0]), (int(x[1]), float(x[2])))).groupByKey().mapValues(list)
	corpus = parsedData.map(lambda x: [x[0], Vectors.sparse(n_vcb, x[1])])

	ldaModel = LDA.train(corpus, k=n)

	print "Learned topics (as distributions over vocab of " + str(ldaModel.vocabSize()) + " words):"
	print ldaModel.describeTopics(maxTermsPerTopic=20)

# to do: calculate document topics
def docTopics(corpus, topicMatrix):
	# for each topic
	# sum probablity of words in corpus
        # normalize so that probility of topics sum to 1
	return 0


def main():
	#vocabulary = create_vocabulary('lda/temp/')
	
	#bag_of_ngrams('lda/temp/', vocabulary)
     	
	filepath = "lda/docword.nyt.txt"
	filepath = "lda/nips.txt"
        topic_n = 10
	runLDA(filepath, topic_n)
                                                                                                            
if __name__ == "__main__":
	main()

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
