<?php

require_once __DIR__ . '/../../../vendor/autoload.php';

// Phpml's LeastSquares regressor finds the linear pattern hidden in the
// data, so the rows do not need to be in any particular order. The sum of
// squared errors it minimizes runs over every sample, which makes the
// learned coefficients independent of row order.
use Phpml\Regression\LeastSquares;

/*
 * 1. INSTANTIATING THE REGRESSOR
 *
 * LeastSquares has no required constructor arguments. During train(), it
 * calculates the intercept and coefficients that minimize squared errors.
 */
$regressor = new LeastSquares();

/*
 * 2. UNORDERED FEATURE VECTORS (INPUTS)
 *
 * A courier company logs every delivery with two numeric features:
 * [distance in km, number of stops]. The rows below come straight from a
 * log file, so they are NOT in any meaningful order — some long trips
 * appear before short ones, and stops vary independently. Each row is one
 * observation, and the pattern lives in the relationship between the
 * columns, not in the row sequence.
 */
$samples = [
    [12, 5],  // long haul, many stops
    [2, 1],
    [7, 3],
    [15, 2],  // very long haul, few stops — out of order on purpose
    [4, 2],
    [9, 6],
    [3, 4],
    [10, 1],
    [6, 2],
    [8, 4],
];

/*
 * 3. CONTINUOUS TARGETS
 *
 * Each target is the delivery time in minutes recorded for the sample at
 * the same index. The real operation follows the pattern:
 *   time = 10 + 2 * distance + 3 * stops
 * e.g. [12, 5] => 10 + 24 + 15 = 49. The model must recover this pattern
 * from shuffled rows, the way it would from a real unordered log.
 */
$targets = [49, 17, 33, 42, 24, 46, 28, 33, 28, 38];

/*
 * 4. TRAINING / FITTING THE PLANE
 *
 * PHP-ML estimates the model:
 *   prediction = intercept + coef_distance * km + coef_stops * stops
 *
 * Row order is irrelevant here: the normal equation sums over all samples,
 * so the learned values should be approximately:
 *   intercept = 10, coef_distance = 2, coef_stops = 3.
 */
$regressor->train($samples, $targets);

echo 'Learned intercept: ' . $regressor->getIntercept() . PHP_EOL;
echo 'Learned coefficient for distance: ' . $regressor->getCoefficients()[0] . PHP_EOL;
echo 'Learned coefficient for stops: ' . $regressor->getCoefficients()[1] . PHP_EOL;

/*
 * 5. PREDICTION ON NEW INPUTS
 *
 * These deliveries were not part of the log. The regressor substitutes
 * each feature vector into the learned equation, proving the pattern was
 * recognized even though the training rows were unordered.
 */
$testSamples = [
    [5, 3],   // expected: 10 + 10 + 9  = 29
    [14, 2],  // expected: 10 + 28 + 6  = 44
    [8, 5],   // expected: 10 + 16 + 15 = 41
];

$predictions = $regressor->predict($testSamples);

/*
 * 6. OUTPUT RESULTS
 */
foreach ($testSamples as $index => $sample) {
    echo 'Delivery [distance: ' . $sample[0] . ' km, stops: ' . $sample[1]
        . '] => Predicted Time: ' . round($predictions[$index], 1) . ' min'
        . PHP_EOL;
}
