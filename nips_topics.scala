import scala.collection.mutable
import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD


val data = sc.textFile("lda/nips.txt")

val n_docs = data.take(1)(0)
val n_voc = data.take(2)(1)
val n_wds = data.take(3)(2)

val doc_map = data.map(x => x.split(" ")).filter(x => x.length > 2).map(x => (x(0).toInt, (x(1).toInt, x(2).toDouble))).groupByKey()

val documents: RDD[(Long, Vector)] = doc_map.map(x => (x._1, Vectors.sparse(12419, x._2.map(_._1).toArray, x._2.map(_._2).toArray)))

val numTopics = 10

val lda = new LDA().setK(numTopics).setMaxIterations(100)

val ldaModel = lda.run(documents)

val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)

val vocabulary = sc.textFile("lda/nips_voc.txt")

val vocabArray = vocabulary.collect

topicIndices.foreach { case (terms, termWeights) =>
     println("TOPIC:")
     terms.zip(termWeights).foreach { case (term, weight) =>
     println(s"${vocabArray(term.toInt)}\t$weight")
     }
     println()
}
   