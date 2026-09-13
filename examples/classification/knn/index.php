<?php

require_once __DIR__ . '/../../../vendor/autoload.php';

// Phpml implements k-Nearest Neighbors as a lazy, instance-based classifier.
// It stores the training samples and classifies a new sample by comparing it
// with the closest samples in feature space.
use Phpml\Classification\KNearestNeighbors;
use Phpml\Math\Distance\Euclidean;

/*
 * 1. INSTANTIATING THE CLASSIFIER
 *
 * The first argument is k, the number of neighbors used to vote.
 *
 * We use k = 3 (an odd number) to reduce the chance of a tie in this
 * two-class example. Euclidean distance is passed explicitly; it is also
 * PHP-ML's default distance metric.
 */
$classifier = new KNearestNeighbors(3, new Euclidean());

/*
 * 2. FEATURE VECTORS (SAMPLES)
 *
 * Each row is one observation with two numeric features: [feature 1, feature 2].
 * The first four observations form class A's cluster and the last four form
 * class B's cluster.
 */
$samples = [
    [3, 6],
    [3, 4],
    [4, 3],
    [4, 4],
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],
];

/*
 * 3. TARGET LABELS (GROUND TRUTH)
 *
 * Each label corresponds to the sample at the same index. k-NN needs these
 * labels because prediction is based on the majority class of nearby samples.
 */
$labels = [
    'A', // [3, 6]
    'A', // [3, 4]
    'A', // [4, 3]
    'A', // [4, 4]
    'B', // [1, 1]
    'B', // [1, 2]
    'B', // [2, 1]
    'B', // [2, 2]
];

/*
 * 4. TRAINING / STORING THE DATA
 *
 * Unlike SVM, k-NN does not learn weights or build a decision boundary here.
 * train() keeps the samples and labels in memory for use during prediction.
 */
$classifier->train($samples, $labels);

/*
 * 5. INFERENCE / PREDICTION
 *
 * For each query, PHP-ML:
 *   1. Calculates its Euclidean distance from every training sample.
 *   2. Selects the three smallest distances.
 *   3. Counts the labels of those neighbors.
 *   4. Returns the label with the most votes.
 */
$testSamples = [
    [3, 5], // Close to class A => expected: A
    [1, 1], // Matches a class B sample => expected: B
];

$predictions = $classifier->predict($testSamples);

/*
 * 6. OUTPUT RESULTS
 */
foreach ($testSamples as $index => $sample) {
    echo 'Sample [' . implode(', ', $sample) . '] => Predicted Class: '
        . $predictions[$index] . PHP_EOL;
}
