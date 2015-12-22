/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.clustering

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import thinkbayes.Cdf
import scala.util.Random

class GMeans extends KMeans {
  def recursivelySplit(data: RDD[Vector], maxIterations: Int, runs: Int): Array[Vector] = {
    val model = new KMeans().setK(1)
      .setMaxIterations(maxIterations)
      .setRuns(runs)
      .setInitializationMode(KMeans.RANDOM)
      .run(data)
    val centers: Array[Vector] = model.clusterCenters
    if (centers.length != 1){
      return null
    }
    val center: Vector = centers { 0}
    val greater: Boolean = ADTest(center, data)
    var result: Array[Vector] = Array()
    if (greater) {
      var model: KMeansModel = splitWithTwoCentres(data, maxIterations, runs)
      val collect: Array[Vector] = data.collect()
      val cluster1: RDD[Vector] = data.filter(model.predict(_) == 0)
      val cluster2: RDD[Vector] = data.filter(model.predict(_) == 1)
      var result11: Array[Vector] = recursivelySplit(cluster1, maxIterations, runs)
      var result12: Array[Vector] = recursivelySplit(cluster2, maxIterations, runs)
      result = result11 ++ result12
    }
    else {
      result = centers
    }
    result
  }

  def train(data: RDD[Vector], maxIterations: Int, runs: Int): Array[Vector] = {
    val clusterCentres: Array[Vector] = recursivelySplit(data, maxIterations, runs)
    println(clusterCentres.toSeq)
    clusterCentres
  }

  def splitWithTwoCentres(data: RDD[Vector], maxIterations: Int, runs: Int): KMeansModel = {
    val model = new KMeans().setK(2).setMaxIterations(maxIterations).setRuns(runs)
      .setInitializationMode(KMeans.RANDOM).run(data)//
    model
  }

  def dotProduct(doubles: List[Double], doubles1: List[Double]): Double = {
    var product: List[Double] = List()
    for (i <- 0 to doubles.length - 1) {
      product :::= List(doubles(i) * doubles1(i))
    }
    product.sum
  }

  def subtractTwoVectors(value: List[Double], value1: List[Double]): List[Double] = {
    var result: List[Double] = List();
    var i1: Int = value.length - 1
    for (i <- 0 to i1) {
      result :::= List(value(i) - value1(i))
    }
    result
  }

  def magnitudeOf(point: List[Double]): Double = {
    math.sqrt(dotProduct(point, point))
  }

  def projectSingleVector(ref: List[Double], doubles1: List[Double], doubles2: List[Double]):
  Double = {//
    val d: Double = dotProduct(doubles1, doubles2) / dotProduct(doubles1, doubles1)
    val intermediate: List[Double] = doubles1.map(x => x * d)
    val result: List[Double] = subtractTwoVectors(intermediate, ref)
    magnitudeOf(result)
  }

  def mean(xs: List[Double]): Double = xs match {
    case Nil => 0.0
    case ys => ys.reduceLeft(_ + _) / ys.size.toDouble
  }

  def stddev(xs: List[Double], avg: Double): Double = xs match {
    case Nil => 0.0
    case ys => math.sqrt((0.0 /: ys) {(a, e) => a + math.pow(e - avg, 2.0)} / xs.size)
  }

  def doShiftingScaling(doubles: List[Double]): List[Double] = {
    val mean1: Double = mean(doubles)
    val dev: Double = stddev(doubles, mean1)
    val toReturn: List[Double] = doubles.map(x => x - mean1).map(x => x / dev)
    toReturn
  }

  def projectDataOntoC(array1: List[Double], array2: List[Double], rdd: RDD[Vector]): List[Double]
  = {//
    var c: List[Double] = subtractTwoVectors(array1, array2)
    var projections: List[Double] = List()
    var collect: Array[Vector] = rdd.collect()
    for (i <- 0 to rdd.count.toInt - 1) {
      var scalar: Double = projectSingleVector(array1, array2, collect(i.toInt).toArray.toList)
      projections :::= List(scalar)
    }
    doShiftingScaling(projections)
  }

  def getCDF(x: List[Double]): Seq[(Double, Double)] = {
    var n: Int = x.length
    var toInitializeCDF: Map[Double, Double] = Map();
    var tupleList: List[(Double, Double)] = List()
    var prob: Double = (1.toFloat / n)
    var m = mean(x)
    var s = stddev(x, x.sum)
    for (i <- 0 to n - 1) {
      if (toInitializeCDF.keySet.exists(_ == x(i))) {
        val maybeDouble: Option[Double] = toInitializeCDF get x(i)
        toInitializeCDF += (x(i) -> (maybeDouble.get + prob))
      } else {toInitializeCDF += (x(i) -> prob)}
      tupleList = tupleList :+(x(i), prob)
    }
    tupleList.toSeq
  }

  def andersonDarlingTest(doubles: List[Double]): Double = {
    var n: Int = doubles.length
    var tupleList = getCDF(doubles)
    val cdf = Cdf(tupleList: _*)
    var sum: Double = 0.0
    for (j <- 0 to n - 1) {
      sum += (2 * j - 1) * (scala.math.log(cdf.prob(doubles(j)))
        + scala.math.log(cdf.prob(doubles(n - j - 1))))}
    -n - (1.0 / (n * 1.0)) * sum
  }

  def ADTest(centre: Vector, data: RDD[Vector]): Boolean = {
    var n = centre.size;
    var fill: List[Double] = Seq.fill(n)(Random.nextInt(2).toDouble).toList
    var centreArray: List[Double] = centre.toArray.toList
    var add: List[Double] = List();
    var subtract: List[Double] = List();
    var i: Int = 0;
    for (i <- 0 to n - 1) {
      subtract :::= List(centreArray(i) - fill(i));
      add :::= List(centreArray(i) + fill(i));
    }
    var dataOntoC: List[Double] = projectDataOntoC(add, subtract, data)
    var A2: Double = andersonDarlingTest(dataOntoC);
    if (A2 > 1.89){
      return true}
    else{return false
    }
  }
}
object GMeans {

  // Initialization mode names
  val RANDOM = "random"
  val K_MEANS_PARALLEL = "k-means||"

  /**
   * Trains a k-means model using the given set of parameters.
   *
   * @param data training points stored as `RDD[Vector]`
   * @param k number of clusters
   * @param maxIterations max number of iterations
   */
  def train(
             data: RDD[Vector],
             k: Int,
             maxIterations: Int): Array[Vector] = {
    new GMeans().train(data, k, maxIterations)
  }
}
