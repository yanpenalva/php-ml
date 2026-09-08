<?php

require_once 'vendor/autoload.php';

use Phpml\Association\Apriori;

$apriori = new Apriori();

$samples =
[
['milk', 'bread','cheese'],
['fish', 'shrimp', 'salmon'],
['eggs', 'bacon', 'toast'],
['sun', 'clouds', 'rain']
];

$apriori->train($samples, []);

$result = $apriori->predict([['milk', 'bread']]);

foreach ($result[0] as $association) {
    echo implode(', ', $association) . PHP_EOL;
}
