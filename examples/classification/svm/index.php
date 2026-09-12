<?php

require_once __DIR__ . '/../../../vendor/autoload.php';

// Phpml provides Support Vector Classification via the SVC class,
// which is a wrapper around the libsvm library.
use Phpml\Classification\SVC;

/*
 * 1. INSTANTIATING THE CLASSIFIER
 *
 * SVM Core Concept:
 * An SVM aims to find an optimal "hyperplane" (a boundary) that separates classes
 * with the maximum possible margin (the distance between the boundary and the closest data points).
 *
 * By default, Phpml's SVC constructor uses:
 *  - Kernel: RBF (Radial Basis Function) — projects non-linear data into higher dimensions.
 *  - Cost parameter ($C = 1.0): Controls the trade-off between maximizing the margin
 *    and minimizing misclassifications (slack tolerance).
 *
 * Example of explicit configuration:
 * $classifier = new SVC(Kernel::RBF, $cost = 1000);
 */
$classifier = new SVC();

/*
 * 2. FEATURE VECTORS (SAMPLES)
 *
 * Each array represents a single data point in a 2-dimensional feature space (x, y).
 *
 * - Cluster 1 (first 4 samples): coordinates are generally around x: 3-4, y: 3-6.
 * - Cluster 2 (next 4 samples): coordinates are clustered around x: 1-2, y: 1-2.
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
 * SVM is a supervised learning algorithm, meaning every sample requires a corresponding label.
 * The indexes of $samples and $labels must map 1:1.
 */
$labels = [
    'A', // maps to [3, 6]
    'A', // maps to [3, 4]
    'A', // maps to [4, 3]
    'A', // maps to [4, 4]
    'B', // maps to [1, 1]
    'B', // maps to [1, 2]
    'B', // maps to [2, 1]
    'B', // maps to [2, 2]
];

/*
 * 4. TRAINING / MODEL FITTING
 *
 * What happens internally during train():
 * 1. The solver identifies the "Support Vectors" — the critical boundary points
 *    closest to the opposing class that define the separation corridor.
 * 2. It computes the mathematical weights and bias that maximize the margin corridor.
 * 3. Points far away from the boundary are discarded; only the support vectors dictate the decision function.
 */
$classifier->train($samples, $labels);

/*
 * 5. INFERENCE / PREDICTION
 *
 * The trained decision boundary determines which side of the hyperplane a new sample lands on:
 * - [3, 5]: Positioned closely within the 'A' cluster region (predicts 'A').
 * - [2, 2]: Exactly matches one of the known 'B' coordinates (predicts 'B').
 */
$result = $classifier->predict([
    [3, 5],
    [2, 2],
]);

/*
 * 6. OUTPUT RESULTS
 */
foreach ($result as $label) {
    echo $label . PHP_EOL;
}
