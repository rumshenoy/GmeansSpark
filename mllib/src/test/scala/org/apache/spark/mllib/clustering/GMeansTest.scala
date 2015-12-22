package org.apache.spark.mllib.clustering
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.scalatest.FunSuite

/**
 * Created by ramyashenoy on 5/10/15.
 */
class GMeansTest extends FunSuite with MLlibTestSparkContext{
  test("check"){
    val data = sc.parallelize(Array(
      Vectors.dense(1.0, 2.0, 9.0),
      Vectors.dense(2.0, 3.0, 5.0),
      Vectors dense(1.5, 2.5, 5.0),
      Vectors.dense(2.0, 3.0, 4.5),
      Vectors.dense(15.0, 15.0, 30.0),
      Vectors.dense(13.0, 12.0, 31.0),
      Vectors.dense(16.0, 11.0, 30.0),
      Vectors.dense(14.0, 13.0, 32.0),
      Vectors.dense(15.0, 10.0, 30.0),
      Vectors.dense(10.0, 16.0, 27.0)
    ))
    var gmeans = new GMeans();
    val centres: Array[Vector] = gmeans.train(data, 1, 1)
    assert(centres.length == 2)
  }

}
