# PHP-ML Study Lab

This is my personal Machine Learning study repository using PHP. It is a small space for reproducing studied examples, implementing experiments, documenting concepts, and understanding algorithms beyond library APIs.

## Stack

- PHP 8.4.25 (the version currently installed in the environment)
- Composer
- [PHP-ML](https://github.com/php-ai/php-ml) 0.10.0

## Installation

```bash
composer install
```

## Run the current example

```bash
php index.php
```

The current example uses the Apriori association-rule learner with a small set of transactions.

## Studied topics

- Association rules and the Apriori algorithm.
- Transactions, items, itemsets, support, and confidence.
- Candidate generation, pruning, and rule generation.

## Organization

- `index.php` — current executable study example.
- `notes/` — concise notes about concepts and algorithms.
- `examples/` — additional focused reproductions, added as needed.
- `experiments/` — personal variations and investigations, added as needed.
- `composer.json` and `composer.lock` — project dependencies.

Course examples may be reproduced here as study references, but this repository is not official course material and is not structurally coupled to any course. Personal experiments and notes are kept independent so the lab can evolve with my studies.
