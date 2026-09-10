# Classification Analysis in PHP-ML

**Classification Analysis** is a fundamental branch of **supervised machine learning** where the goal is to predict a discrete categorical label (class) for an input observation based on patterns learned from previously labeled training data.

This document serves as the complete guide to Classification Analysis in PHP-ML, detailing its theoretical foundations, mathematical metrics, and in-depth guides for the three primary classification algorithms supported by the library:

1. **Support Vector Classification (SVC / SVM)**
2. **k-Nearest Neighbors (k-NN)**
3. **Naive Bayes**

---

## 1. Core Concepts & Mathematical Foundation

Unlike unsupervised Association Analysis (which finds arbitrary co-occurring itemsets without targets), Classification learns a mapping function from input feature vectors to discrete target classes.

### 1.1 Key Terminology

- **Feature Vector ($x$):** A $d$-dimensional vector of numerical or categorical attributes representing an observation:
  $$x = [x_1, x_2, \dots, x_d] \in \mathbb{R}^d$$
- **Target Label ($y$):** The ground-truth categorical class assigned to an observation:
  $$y \in \mathcal{C} = \{c_1, c_2, \dots, c_K\}$$
    - **Binary Classification:** $|\mathcal{C}| = 2$ (e.g., Spam vs. Ham, Fraud vs. Legitimate).
    - **Multi-class Classification:** $|\mathcal{C}| > 2$ (e.g., Digit recognition $0 \dots 9$, Sentiment: Positive, Neutral, Negative).
- **Training Set ($D_{\text{train}}$):** The set of labeled pairs used to fit the model:
  $$D = \{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \dots, (x^{(N)}, y^{(N)})\}$$
- **Decision Boundary:** The hypersurface that partitions the feature space into regions assigned to different classes.
- **Hypothesis Function ($h_\theta(x)$):** The learned model function that predicts $\hat{y} = h(x)$.

---

### 1.2 Evaluation Metrics

To evaluate classification performance on unseen test data, predictions are compared against ground-truth labels using a **Confusion Matrix**:

|                                        | Actual Positive ($y = 1$)              | Actual Negative ($y = 0$)             |
| :------------------------------------- | :------------------------------------- | :------------------------------------ |
| **Predicted Positive ($\hat{y} = 1$)** | True Positive (**TP**)                 | False Positive (**FP**, Type I Error) |
| **Predicted Negative ($\hat{y} = 0$)** | False Negative (**FN**, Type II Error) | True Negative (**TN**)                |

#### 1. Accuracy

Overall proportion of correct predictions across all classes:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- _Limitation:_ Misleading on imbalanced datasets (e.g., $99\%$ negative class).

#### 2. Precision

Of all samples predicted as positive, how many were actually positive:
$$\text{Precision} = \frac{TP}{TP + FP}$$

- High precision is critical when false positives are costly (e.g., spam filter moving important emails to spam).

#### 3. Recall (Sensitivity)

Of all actual positive samples, how many were successfully detected:
$$\text{Recall} = \frac{TP}{TP + FN}$$

- High recall is critical when false negatives are dangerous (e.g., cancer diagnosis, fraud detection).

#### 4. F1-Score

The harmonic mean of precision and recall:
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

---

## 2. Algorithm Guides: What Runs Behind the Scenes

### 2.1 Support Vector Classification (SVC / SVM)

The **Support Vector Machine** (Cortes & Vapnik, 1995) finds the optimal hyperplane that separates classes with the **maximum geometric margin**.

```text
       Feature 2
           ^           Class +1
           |         o      o
           |       o    o [Support Vector]
           |     ---------------------  Hyperplane: w^T x + b = +1
           |          /       /
           |         / Margin /   Optimal Separating Hyperplane:
           |        /    M   /    w^T x + b = 0
           |       /       /
           |     ---------------------  Hyperplane: w^T x + b = -1
           |        x   [Support Vector]
           |      x   x    Class -1
           +------------------------------------> Feature 1
```

#### Maximum Margin & Support Vectors

A linear hyperplane is defined as:
$$w^T x + b = 0$$
The distance from the hyperplane to the closest data points is $\frac{1}{\|w\|}$. Maximizing this margin is equivalent to minimizing $\frac{1}{2} \|w\|^2$.
The data points that lie directly on the margin boundaries ($w^T x + b = \pm 1$) are called **Support Vectors**. The decision boundary depends _strictly_ on these support vectors; moving any other point does not change the model.

#### Soft Margin & The Cost Parameter ($C$)

Real-world data is rarely linearly separable. Soft Margin SVM introduces slack variables $\xi_i \ge 0$ to allow controlled misclassifications:
$$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i \quad \text{subject to } y_i(w^T x_i + b) \ge 1 - \xi_i$$

- **High $C$:** Penalizes errors heavily. Produces a narrower margin; risks **overfitting**.
- **Low $C$:** More tolerant of misclassifications. Produces a wider margin; risks **underfitting**.

#### The Kernel Trick

When data cannot be separated by a hyperplane in its original space $\mathbb{R}^d$, SVM maps data into a higher-dimensional space $\Phi(x)$ where linear separation is possible. The **Kernel Trick** computes inner products in high-dimensional space without ever explicitly computing $\Phi(x)$:
$$K(u, v) = \langle \Phi(u), \Phi(v) \rangle$$

Supported Kernels in PHP-ML:

1. **Linear:** $K(u, v) = u^T v$
2. **Polynomial:** $K(u, v) = (\gamma u^T v + \text{coef}_0)^{\text{degree}}$
3. **RBF (Radial Basis Function / Gaussian):**
   $$K(u, v) = \exp(-\gamma \|u - v\|^2)$$
    - $\gamma$ (gamma) controls the radius of influence:
        - High $\gamma$: Small radius; each point has high local influence (complex, jagged boundary $\to$ overfitting).
        - Low $\gamma$: Large radius; smooth decision boundary ($\to$ underfitting).
4. **Sigmoid:** $K(u, v) = \tanh(\gamma u^T v + \text{coef}_0)$

---

### 2.2 k-Nearest Neighbors (k-NN)

**k-Nearest Neighbors** (Fix & Hodges, 1951) is a **non-parametric, instance-based (lazy) learner**.

```text
       Feature 2
           ^
           |       Class A (o)
           |      o       o
           |        o   [?] <-- New Query Point
           |          /  |  \
           |        x    x   o
           |      x     Class B (x)
           +------------------------------------> Feature 1
           k=1: Nearest is 'o' -> Predict Class A
           k=3: Neighbors are 2 'x' and 1 'o' -> Predict Class B (Majority Vote)
```

#### Key Characteristics

- **No Training Phase ("Lazy"):** k-NN does not construct an explicit model or decision boundary during `train()`. It simply stores the training dataset in memory.
- **Inference Cost ("Eager Predict"):** All computational work happens during `predict()`, where the distance between the query point and _every stored training instance_ is computed.

#### Distance Metrics

For two points $u, v \in \mathbb{R}^d$:

1. **Euclidean Distance ($L_2$ norm, default):**
   $$d(u, v) = \sqrt{\sum_{i=1}^d (u_i - v_i)^2}$$
2. **Manhattan Distance ($L_1$ norm):**
   $$d(u, v) = \sum_{i=1}^d |u_i - v_i|$$
3. **Chebyshev Distance ($L_\infty$ norm):**
   $$d(u, v) = \max_{i=1 \dots d} |u_i - v_i|$$
4. **Minkowski Distance ($L_p$ norm):**
   $$d(u, v) = \left(\sum_{i=1}^d |u_i - v_i|^p\right)^{\frac{1}{p}}$$

#### Decision Rule: Majority Vote

1. Compute distance from query point $x$ to all training samples.
2. Select the $k$ nearest training samples.
3. Tally class occurrences among the $k$ neighbors.
4. Assign the class with the highest vote count:
   $$\hat{y} = \arg\max_{c \in \mathcal{C}} \sum_{i \in N_k(x)} \mathbb{I}(y_i = c)$$

> [!TIP]
> **Choosing $k$ & Feature Scaling:**
>
> - Small $k$ (e.g., $k=1$): Very low bias, high variance; highly sensitive to noise and outliers.
> - Large $k$: High bias, low variance; smooth boundary, but may dilute minority classes.
> - **Odd numbers** (3, 5, 7) are preferred in binary classification to break ties.
> - **Feature Scaling is mandatory:** Features with large numerical ranges (e.g., Salary: $50,000) will dominate features with small ranges (e.g., Age: 30) in distance calculations.

---

### 2.3 Naive Bayes

**Naive Bayes** (based on Thomas Bayes' theorem) is a probabilistic generative classifier.

#### Bayes' Theorem

Given a feature vector $X = [x_1, x_2, \dots, x_d]$, we seek the posterior probability of class $y$:
$$P(y \mid X) = \frac{P(X \mid y) P(y)}{P(X)}$$

#### The "Naive" Conditional Independence Assumption

Computing joint probability $P(x_1, x_2, \dots, x_d \mid y)$ directly is intractable. Naive Bayes makes the simplifying assumption that **all features are conditionally independent given the class**:
$$P(X \mid y) = \prod_{i=1}^d P(x_i \mid y)$$

Since $P(X)$ is constant for all classes, the classification rule maximizes the posterior (MAP rule):
$$\hat{y} = \arg\max_{y \in \mathcal{C}} \left( P(y) \prod_{i=1}^d P(x_i \mid y) \right)$$

#### In Log-Space (Preventing Underflow)

Multiplying many small probabilities causes floating-point underflow. Taking the logarithm converts products to sums:
$$\hat{y} = \arg\max_{y \in \mathcal{C}} \left( \log P(y) + \sum_{i=1}^d \log P(x_i \mid y) \right)$$

#### How PHP-ML Handles Feature Types

PHP-ML's `NaiveBayes` automatically inspects feature values:

1. **Continuous / Numeric Features (Gaussian Naive Bayes):**
   Assumes features follow a Normal (Gaussian) distribution:
   $$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma_{y,i}^2}} \exp\left( -\frac{(x_i - \mu_{y,i})^2}{2\sigma_{y,i}^2} \right)$$
   Where $\mu_{y,i}$ is the sample mean and $\sigma_{y,i}$ is the standard deviation for feature $i$ given class $y$.
2. **Discrete / Nominal Features (Categorical Naive Bayes):**
   Computes relative frequency of each category within the class:
   $$P(x_i = v \mid y) = \frac{\text{count}(x_i = v \text{ in class } y)}{\text{total samples in class } y}$$
   An $\epsilon$ ($10^{-10}$) is added to avoid zero probabilities for unseen categories.

---

## 3. Under the Hood in PHP-ML

All classifiers in PHP-ML implement the `Phpml\Classification\Classifier` interface:

```php
namespace Phpml\Classification;

interface Classifier
{
    public function train(array $samples, array $targets): void;
    public function predict(array $samples);
}
```

### 3.1 Class Architecture Overview

```text
                     +-----------------------------------+
                     | Phpml\Classification\Classifier   |
                     +-----------------------------------+
                                       ^
                                       | implements
         +-----------------------------+-----------------------------+
         |                             |                             |
+------------------+         +--------------------+        +---------------------+
|       SVC        |         |  KNearestNeighbors |        |     NaiveBayes      |
+------------------+         +--------------------+        +---------------------+
| Wraps libsvm CLI |         | Memory-based lazy  |        | Probabilistic model |
| Uses Kernel      |         | Computes Distances |        | Gaussian & Nominal  |
+------------------+         +--------------------+        +---------------------+
```

---

### 3.2 SVC Internals (`Phpml\Classification\SVC`)

1. **`SVC` extends `SupportVectorMachine`:**
    - PHP-ML does not compute SVM quadratic programming in pure PHP. Instead, it delegates to a compiled **`libsvm`** binary located in `vendor/php-ai/php-ml/bin/libsvm/`.
2. **`train($samples, $targets)`:**
    - Serializes training data into libsvm format using `DataTransformer::trainingSet()`.
    - Writes the dataset to a temporary file in `var/` (`uniqid('phpml', true)`).
    - Executes the external libsvm training binary via `exec()`.
    - Reads the generated model file into memory (`$this->model`) and unlinks temporary files.
3. **`predict($samples)`:**
    - Writes query samples to a temporary file.
    - Runs `svm-predict` binary via `exec()`.
    - Maps predicted labels back to PHP-ML format using `DataTransformer::predictions()`.

---

### 3.3 k-NN Internals (`Phpml\Classification\KNearestNeighbors`)

1. **`train($samples, $targets)`:**
    - Uses the `Trainable` trait. Simply merges `$samples` into `$this->samples` and `$targets` into `$this->targets`.
2. **`predictSample(array $sample)`:**
    - Calls `kNeighborsDistances($sample)`:
        ```php
        foreach ($this->samples as $index => $neighbor) {
            $distances[$index] = $this->distanceMetric->distance($sample, $neighbor);
        }
        asort($distances);
        return array_slice($distances, 0, $this->k, true);
        ```
    - Tallies votes using `array_combine` and increments counts for nearest classes.
    - Sorts with `arsort($predictions)` and returns `key($predictions)`.

---

### 3.4 Naive Bayes Internals (`Phpml\Classification\NaiveBayes`)

1. **`train($samples, $targets)`:**
    - Calculates prior probabilities $P(y) = \frac{N_y}{N}$.
    - Scans each feature column via `calculateStatistics()`:
        - Tests `is_numeric()` on values.
        - If nominal: computes relative frequencies in `$this->discreteProb`.
        - If numeric: computes `Mean::arithmetic` and `StandardDeviation::population`.
2. **`predictSample(array $sample)`:**
    - For each class, starts with prior probability $P(y)$.
    - Sums log-probabilities of all features using Gaussian PDF:
        ```php
        $pdf = -0.5 * log(2.0 * M_PI * $std * $std);
        $pdf -= 0.5 * (($value - $mean) ** 2) / ($std * $std);
        ```
    - Picks the class with highest posterior probability.

---

## 4. Parameter Classes & Configuration Scenarios

### 4.1 Input Data Formats (Mandatory for all Classifiers)

Unlike Association Analysis (which receives raw transaction baskets), Classification requires **strictly structured supervised inputs**:

```php
$classifier->train(array $samples, array $targets);
```

#### 1. Feature Matrix (`$samples`)

- **Structure:** `array<int, array<int, float|int|string>>`
- A **2-dimensional array** where each row represents one observation, and each column represents a feature attribute:
    ```php
    // Correct format: 2D Matrix (N samples x D features)
    $samples = [
        [3.0, 6.0],
        [3.0, 4.0],
        [1.0, 1.0],
        [2.0, 1.0],
    ];
    ```

> [!CAUTION]
> **Common Beginner Trap (Nested Arrays):**
> Passing a 3D array like `[[[3, 6], [1, 1]]]` will fail or produce incorrect results. Ensure each sample is a flat list of feature values `[3.0, 6.0, 1.0, 1.0]`.

#### 2. Label Vector (`$targets`)

- **Structure:** `array<int, string|int>`
- A **1-dimensional array** of length $N$, matching rows in `$samples`:
    ```php
    $targets = ['A', 'A', 'B', 'B'];
    ```

---

### 4.2 Algorithm Constructors & Hyperparameters

#### 1. SVC (`Phpml\Classification\SVC`)

```php
use Phpml\Classification\SVC;
use Phpml\SupportVectorMachine\Kernel;

$classifier = new SVC(
    int $kernel = Kernel::RBF,
    float $cost = 1.0,
    int $degree = 3,
    ?float $gamma = null,
    float $coef0 = 0.0,
    float $tolerance = 0.001,
    int $cacheSize = 100,
    bool $shrinking = true,
    bool $probabilityEstimates = false
);
```

| Hyperparameter          | Type     | Default       | Description & Guidelines                                                                |
| :---------------------- | :------- | :------------ | :-------------------------------------------------------------------------------------- |
| `$kernel`               | `int`    | `Kernel::RBF` | `Kernel::LINEAR`, `Kernel::POLYNOMIAL`, `Kernel::RBF`, or `Kernel::SIGMOID`.            |
| `$cost` ($C$)           | `float`  | `1.0`         | Regularization penalty. Higher = strict margin (overfitting risk); Lower = soft margin. |
| `$degree`               | `int`    | `3`           | Degree for `Kernel::POLYNOMIAL`.                                                        |
| `$gamma` ($\gamma$)     | `?float` | `null`        | Kernel coefficient for RBF/Poly/Sigmoid. Defaults to $1 / \text{num\_features}$.        |
| `$coef0`                | `float`  | `0.0`         | Independent term in kernel functions.                                                   |
| `$tolerance`            | `float`  | `0.001`       | Stopping criterion tolerance.                                                           |
| `$probabilityEstimates` | `bool`   | `false`       | Whether to train for probability output.                                                |

#### 2. k-NN (`Phpml\Classification\KNearestNeighbors`)

```php
use Phpml\Classification\KNearestNeighbors;
use Phpml\Math\Distance\Euclidean;
use Phpml\Math\Distance\Manhattan;
use Phpml\Math\Distance\Chebyshev;
use Phpml\Math\Distance\Minkowski;

$classifier = new KNearestNeighbors(
    int $k = 3,
    ?Distance $distanceMetric = null
);
```

| Hyperparameter    | Type       | Default           | Description & Guidelines                                                                                        |
| :---------------- | :--------- | :---------------- | :-------------------------------------------------------------------------------------------------------------- |
| `$k`              | `int`      | `3`               | Number of nearest neighbors to consider. Use odd numbers for binary tasks.                                      |
| `$distanceMetric` | `Distance` | `new Euclidean()` | Distance metric instance: `new Euclidean()`, `new Manhattan()`, `new Chebyshev()`, or `new Minkowski($lambda)`. |

#### 3. Naive Bayes (`Phpml\Classification\NaiveBayes`)

```php
use Phpml\Classification\NaiveBayes;

$classifier = new NaiveBayes();
```

- Requires **no constructor arguments**. Feature types and statistical distributions are deduced automatically during `train()`.

---

## 5. Handling Predictions & Responses

The `Predictable` trait powers `predict()` across all classifiers, enabling both **single sample** and **batch prediction**.

### 5.1 Single Sample Query

Pass a 1D array of feature values:

```php
$prediction = $classifier->predict([3.2, 5.8]);
// Return: single class label (e.g., 'A')
```

### 5.2 Batch Query

Pass a 2D array of multiple feature vectors:

```php
$predictions = $classifier->predict([
    [3.2, 5.8],
    [1.1, 0.9],
]);
// Return: 1D array of class labels matching the query order:
// [0 => 'A', 1 => 'B']
```

### 5.3 Contrast with Association Analysis Output

| Dimension                 | Association Analysis (`Apriori`)       | Classification Analysis (`SVC`, `k-NN`, `NB`) |
| :------------------------ | :------------------------------------- | :-------------------------------------------- |
| **Prediction Input**      | Item set query (e.g., `[['milk']]`)    | Feature vector (e.g., `[3.2, 5.8]`)           |
| **Output Type**           | Nested 3D array of itemsets            | Single label or 1D array of labels            |
| **Output Meaning**        | Discovered co-occurring consequents    | Predicted class membership                    |
| **Response Multiplicity** | Variable (0, 1, or many rules matched) | Exact: 1 class predicted per input sample     |

---

## 6. End-to-End Practical Comparison Example

Below is a complete, runnable script training and comparing **SVC**, **k-NN**, and **Naive Bayes** on the exact same dataset:

```php
<?php

require_once __DIR__ . '/../../vendor/autoload.php';

use Phpml\Classification\SVC;
use Phpml\Classification\KNearestNeighbors;
use Phpml\Classification\NaiveBayes;
use Phpml\SupportVectorMachine\Kernel;
use Phpml\Math\Distance\Euclidean;

// 1. Dataset: 2D numeric features (X1, X2) mapped to 2 classes ('A' and 'B')
// Class A represents upper-right cluster, Class B represents lower-left cluster
$samples = [
    [3.0, 6.0],
    [3.0, 4.0],
    [4.0, 3.0],
    [4.0, 4.0],
    [1.0, 1.0],
    [1.0, 2.0],
    [2.0, 1.0],
    [2.0, 2.0],
];

$labels = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'];

// 2. Initialize the 3 classifiers
$classifiers = [
    'SVC (RBF Kernel)' => new SVC(Kernel::RBF, 1.0),
    'k-NN (k=3, Euclidean)' => new KNearestNeighbors(3, new Euclidean()),
    'Naive Bayes' => new NaiveBayes(),
];

// 3. Train all classifiers
foreach ($classifiers as $name => $clf) {
    $clf->train($samples, $labels);
}

// 4. Test on unseen observations
$testSamples = [
    [3.5, 4.5], // Expected: 'A'
    [1.5, 1.2], // Expected: 'B'
    [2.5, 2.5], // Boundary point between clusters
];

echo "=== CLASSIFIER COMPARISON RESULTS ===\n\n";

foreach ($classifiers as $name => $clf) {
    echo "--- {$name} ---\n";
    $predictions = $clf->predict($testSamples);

    foreach ($testSamples as $i => $sample) {
        $sampleStr = implode(', ', $sample);
        echo "Sample [{$sampleStr}] => Predicted Class: '{$predictions[$i]}'\n";
    }
    echo "\n";
}
```

### Expected Output Summary

```text
=== CLASSIFIER COMPARISON RESULTS ===

--- SVC (RBF Kernel) ---
Sample [3.5, 4.5] => Predicted Class: 'A'
Sample [1.5, 1.2] => Predicted Class: 'B'
Sample [2.5, 2.5] => Predicted Class: 'B'

--- k-NN (k=3, Euclidean) ---
Sample [3.5, 4.5] => Predicted Class: 'A'
Sample [1.5, 1.2] => Predicted Class: 'B'
Sample [2.5, 2.5] => Predicted Class: 'B'

--- Naive Bayes ---
Sample [3.5, 4.5] => Predicted Class: 'A'
Sample [1.5, 1.2] => Predicted Class: 'B'
Sample [2.5, 2.5] => Predicted Class: 'B'
```

---

## 7. Comparison Matrix & Decision Guide

| Criterion                     | Support Vector Machine (`SVC`)                                   | k-Nearest Neighbors (`k-NN`)                                       | Naive Bayes (`NaiveBayes`)                                                  |
| :---------------------------- | :--------------------------------------------------------------- | :----------------------------------------------------------------- | :-------------------------------------------------------------------------- | ----------- | --------- |
| **Learning Paradigm**         | Eager (optimization via libsvm)                                  | Lazy (instance-based, no training)                                 | Eager (statistical parameter estimation)                                    |
| **Training Complexity**       | $\mathcal{O}(N^2 \cdot d)$ to $\mathcal{O}(N^3 \cdot d)$         | $\mathcal{O}(1)$ (stores data in memory)                           | $\mathcal{O}(N \cdot d)$ (single pass)                                      |
| **Inference (Predict) Speed** | Fast: depends on number of SVs                                   | Slow: $\mathcal{O}(N \cdot d)$ per query                           | Extremely fast: $\mathcal{O}(                                               | \mathcal{C} | \cdot d)$ |
| **Memory Footprint**          | Moderate (stores SVs & model parameters)                         | High (stores entire training set)                                  | Low (stores only means, stds, priors)                                       |
| **Non-linear Boundaries**     | Yes (via RBF/Poly kernels)                                       | Yes (arbitrary local shapes)                                       | Moderate (assumes Gaussian/independent features)                            |
| **High Dimensionality**       | Excellent                                                        | Degrades (Curse of Dimensionality)                                 | Good (handles many independent features)                                    |
| **Feature Scaling**           | **Mandatory**                                                    | **Mandatory**                                                      | Not required for Gaussian features                                          |
| **Interpretability**          | Low (Black-box kernel space)                                     | High (Explainable via closest neighbors)                           | High (Direct probabilistic posteriors)                                      |
| **Best Used When**            | Medium-sized tabular data needing complex non-linear separation. | Small datasets where patterns depend on local geometric proximity. | Text classification, spam filtering, or baseline fast probabilistic models. |
