<?php

require_once __DIR__ . '/../../../vendor/autoload.php';

// Phpml provides Naive Bayes as a probabilistic classifier.
// It estimates how likely each class is for a sample and returns the class
// with the highest posterior probability.
use Phpml\Classification\NaiveBayes;

/*
 * 1. INSTANTIATING THE CLASSIFIER
 *
 * NaiveBayes does not require constructor arguments. During train(), PHP-ML
 * calculates class priors and feature statistics automatically. Numeric
 * features are modeled with a Gaussian distribution.
 */
$classifier = new NaiveBayes();

/*
 * 2. FEATURE VECTORS (SAMPLES)
 *
 * Each row is one observation with two numeric features. The observations
 * form two groups: class A is generally near [3, 4] and class B near [1, 1].
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
 * Labels must have the same order and length as the samples array.
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
 * 4. TRAINING / ESTIMATING PROBABILITIES
 *
 * PHP-ML calculates, for each class:
 *   - the prior probability P(class);
 *   - the mean and standard deviation of every numeric feature.
 *
 * Naive Bayes assumes features are conditionally independent given a class,
 * so their probabilities can be combined when a query is classified.
 */
$classifier->train($samples, $labels);

/*
 * 5. INFERENCE / PREDICTION
 *
 * For each query, PHP-ML evaluates the probability of every class using:
 *   P(class | features) proportional to P(class) * product(P(feature | class))
 *
 * The class with the greatest resulting score is returned. These queries show
 * different situations rather than repeating the training samples:
 *   - [3.5, 5.0] is inside the class A region.
 *   - [1.5, 1.5] is inside the class B region.
 *   - [2.5, 3.0] is closer to the boundary between both groups.
 *   - [4.0, 5.0] is a new point near class A, not an exact training sample.
 */
$testSamples = [
    [3.5, 5.0], // Class A region => expected: A
    [1.5, 1.5], // Class B region => expected: B
    [2.5, 3.0], // Boundary-like point; prediction uses both features
    [4.0, 5.0], // New point near class A => expected: A
];

$predictions = $classifier->predict($testSamples);

/*
 * 6. OUTPUT RESULTS
 */
foreach ($testSamples as $index => $sample) {
    echo 'Input [' . implode(', ', $sample) . '] => Predicted Class: '
        . $predictions[$index] . PHP_EOL;
}
