<?php

require_once __DIR__ . '/../../vendor/autoload.php';

use Phpml\Association\Apriori;

$apriori = new Apriori();

$samples = [
    ['milk', 'bread', 'cheese'],
    ['fish', 'shrimp', 'salmon'],
    ['eggs', 'bacon', 'toast'],
    ['sun', 'clouds', 'rain'],
];

$apriori->train($samples, []);

$result = $apriori->predict([['milk'], ['fish']]);
$associations = array_merge(...$result);
foreach ($associations as $association) {
    echo implode(', ', $association) . PHP_EOL;
}
