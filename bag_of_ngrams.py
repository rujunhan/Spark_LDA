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

App_Name = "LDA.PY"
conf = SparkConf().setAppName(App_Name)
sc = SparkContext(conf=conf)

sqlContext = SQLContext(sc)
#hedline = sqlContext.read.format("com.databricks.spark.xml").option("rowTag", "hedline").load("lda/temp/1815836.xml")

#print hedline.take(1)[0][0]

content = sqlContext.read.format("com.databricks.spark.xml").option("rowTag", "body.content").load("lda/temp/1815836.xml")


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

text = process_corpus((" ").join(content.take(1)[0][0][-1][1]),5)
print text
print len(text)
