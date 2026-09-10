<?php

require_once __DIR__ . '/../../../vendor/autoload.php';

use Phpml\Classification\SVC;

$classifier = new SVC();

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

$labels = [
    'A',
    'A',
    'A',
    'A',
    'B',
    'B',
    'B',
    'B',
];

$classifier->train($samples, $labels);

$result = $classifier->predict([
    [3, 5],
    [2, 2],
]);

foreach ($result as $label) {
    echo $label . PHP_EOL;
}
