#### run this line in terminal: spark-submit --packages com.databricks:spark-xml_0.10:0.4.1 testLDA.py !!! 

from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors
from pyspark.sql import SQLContext
import copy
import numpy as np
import nltk
nltk.data.path.append("/home/rjh347/nltk_data/")
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
        # return a rdd of all text corpus
        textfiles = sqlContext.read.format("com.databricks.spark.xml").option("rowTag", "body.content").load(filepath + "*.xml")
        return textfiles.map(lambda x: parse_xml(x)).map(lambda x: process_corpus(x,5))

def create_vocabulary(path):
	vcb = get_corpus(path).flatMap(lambda x: x).distinct().collect()
	return sorted(vcb)
 	
def bag_of_ngrams(text):
	result = text
	return result
	
def main():
	vocabulary = create_vocabulary('lda/temp/')
	print vocabulary

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
