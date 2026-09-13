<?php

require_once __DIR__ . '/../../vendor/autoload.php';

// Phpml's LeastSquares regressor learns a linear relationship between
// numerical feature vectors and continuous numerical targets.
use Phpml\Regression\LeastSquares;

/*
 * 1. INSTANTIATING THE REGRESSOR
 *
 * LeastSquares has no required constructor arguments. During train(), it
 * calculates the intercept and coefficients that minimize squared errors.
 */
$regressor = new LeastSquares();

/*
 * 2. FEATURE VECTORS (INPUTS)
 *
 * Each row contains one feature: [hours studied]. The target below follows
 * the simple relationship score = 10 + 5 * hours, which makes the learned
 * intercept and coefficient easy to understand.
 */
$samples = [
    [1],
    [2],
    [3],
    [4],
    [5],
];

/*
 * 3. CONTINUOUS TARGETS
 *
 * Targets are numeric values, not categorical labels. Each target matches
 * the sample at the same index: [1] => 15, [2] => 20, and so on.
 */
$targets = [15, 20, 25, 30, 35];

/*
 * 4. TRAINING / FITTING THE LINE
 *
 * PHP-ML estimates the model:
 *   prediction = intercept + coefficient * hours
 *
 * For this dataset, the learned values should be approximately:
 *   intercept = 10 and coefficient = 5.
 */
$regressor->train($samples, $targets);

echo 'Learned intercept: ' . $regressor->getIntercept() . PHP_EOL;
echo 'Learned coefficient: ' . $regressor->getCoefficients()[0] . PHP_EOL;

/*
 * 5. PREDICTION ON NEW INPUTS
 *
 * These values were not used during training. The regressor estimates the
 * target by substituting each input into the learned linear equation.
 */
$testSamples = [
    [0],   // Extrapolation below the training range => expected: 10
    [2.5], // Between known samples => expected: 22.5
    [6],   // Extrapolation above the training range => expected: 40
];

$predictions = $regressor->predict($testSamples);

/*
 * 6. OUTPUT RESULTS
 */
foreach ($testSamples as $index => $sample) {
    echo 'Input [' . implode(', ', $sample) . '] => Predicted Score: '
        . $predictions[$index] . PHP_EOL;
}
