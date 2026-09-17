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

## Run the examples

```bash
php examples/association/index.php
php examples/classification/svm/index.php
php examples/regression/index.php
```

## Studied topics

- **Association Analysis:** Apriori algorithm, frequent itemsets, support, confidence, lift, candidate generation, and pruning (`notes/association.md`).
- **Classification Analysis:** Supervised learning, decision boundaries, metrics, and algorithms: Support Vector Classification (SVC/SVM), k-Nearest Neighbors (k-NN), and Naive Bayes (`notes/classification.md`).
- **Regression Analysis:** Continuous-value prediction, least-squares linear regression, Support Vector Regression (SVR), and evaluation metrics (`notes/regression.md`).
- **Clustering Analysis:** Unsupervised grouping, K-Means objective and Lloyd's algorithm, initialization strategies, cluster evaluation, and DBSCAN/FuzzyCMeans positioning (`notes/clustering.md`).

## Organization

- `notes/` — in-depth technical notes and algorithm study guides (`notes/association.md`, `notes/classification.md`, `notes/regression.md`, `notes/clustering.md`).
- `examples/<context>/index.php` — executable examples per study context (`examples/association/`, `examples/classification/`, `examples/grouping/`).
- `experiments/` — personal variations and investigations, added as needed.
- `composer.json` and `composer.lock` — project dependencies.

