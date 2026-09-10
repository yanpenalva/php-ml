# Association Analysis in PHP-ML

**Association Analysis** (also known as _Association Rule Learning_ or _Market Basket Analysis_) is an unsupervised machine learning branch designed to discover strong relationships, co-occurrences, and frequent patterns among variables in large transaction datasets.

This document serves as the complete guide to Association Analysis in PHP-ML, detailing its theoretical foundations, mathematical metrics, algorithmic mechanics, and hands-on usage with the **Apriori** algorithm.

---

## 1. Core Concepts & Mathematical Foundation

In association analysis, data is represented as collections of transactions, rather than labeled feature vectors. The goal is not to predict a target class, but to identify rules that state: _"when itemset $X$ occurs, itemset $Y$ is also likely to occur"_.

### Key Terminology

- **Item ($i$):** An individual entity or token, e.g., `'milk'`, `'bread'`, or `'diaper'`.
- **Transaction ($t$):** A single observed basket or event composed of a set of items: $t \subseteq I$, e.g., `['milk', 'bread', 'butter']`.
- **Dataset ($D$):** The collection of all transactions $D = \{t_1, t_2, \dots, t_N\}$.
- **Itemset ($X$):** A collection of one or more items, e.g., $\{ \text{milk}, \text{bread} \}$. A $k$-itemset is an itemset containing exactly $k$ items.
- **Candidate itemset ($C_k$):** A potentially frequent $k$-itemset generated during level $k$ that must be evaluated against the transaction dataset.
- **Frequent itemset ($L_k$):** An itemset whose support satisfies the minimum support threshold ($\text{support}(X) \ge \text{min\_support}$).
- **Association rule:** An implication of the form:
  $$X \implies Y \quad \text{where } X \subset I, \; Y \subset I, \; X \cap Y = \emptyset, \; X \neq \emptyset, \; Y \neq \emptyset$$
    - $X$ is the **antecedent** (premise / condition).
    - $Y$ is the **consequent** (prediction / associated outcome).

---

### Core Evaluation Metrics

#### 1. Support

Support measures how frequently an itemset or rule appears across the entire dataset.

$$\text{support}(X) = \frac{|\{t \in D \mid X \subseteq t\}|}{|D|}$$

For a rule $X \implies Y$:
$$\text{support}(X \implies Y) = \text{support}(X \cup Y)$$

- **Range:** $[0.0, 1.0]$
- **Meaning:** If $\text{support}(\{\text{milk}, \text{bread}\}) = 0.40$, $40\%$ of all recorded transactions in the database contain both milk and bread.

#### 2. Confidence

Confidence measures how often items in $Y$ appear in transactions that already contain $X$. It represents the conditional probability $P(Y \mid X)$.

$$\text{confidence}(X \implies Y) = \frac{\text{support}(X \cup Y)}{\text{support}(X)} = \frac{|\{t \in D \mid (X \cup Y) \subseteq t\}|}{|\{t \in D \mid X \subseteq t\}|}$$

- **Range:** $[0.0, 1.0]$
- **Meaning:** If $\text{confidence}(\{\text{milk}\} \implies \{\text{bread}\}) = 0.75$, whenever a customer buys milk, there is a $75\%$ probability that they also buy bread.

#### 3. Lift

While PHP-ML filters rules based on support and confidence, **Lift** is an essential diagnostic metric measuring how much more often $X$ and $Y$ occur together than expected if they were statistically independent:

$$\text{lift}(X \implies Y) = \frac{\text{confidence}(X \implies Y)}{\text{support}(Y)} = \frac{\text{support}(X \cup Y)}{\text{support}(X) \times \text{support}(Y)}$$

- $\text{lift} = 1$: $X$ and $Y$ are independent (co-occurrence is due to random chance).
- $\text{lift} > 1$: Positive association (presence of $X$ increases the likelihood of $Y$).
- $\text{lift} < 1$: Negative association (substitutes; presence of $X$ decreases the likelihood of $Y$).

---

## 2. Algorithm Guide: Apriori

The **Apriori algorithm** (Agrawal & Srikant, 1994) is the benchmark algorithm for association analysis.

### 2.1 The Apriori Principle (Anti-monotonicity)

The search space of all possible itemsets is a power set lattice containing $2^{|I|} - 1$ non-empty itemsets. For large vocabularies, brute-force evaluation of all subsets is computationally impossible.

Apriori solves this using the **downward-closure property of support**:

$$X \subseteq Y \implies \text{support}(Y) \le \text{support}(X)$$

> **The Fundamental Axiom:**
> If an itemset is frequent, all of its subsets must also be frequent.
>
> **The Pruning Rule (Contrapositive):**
> If an itemset is **infrequent**, any superset containing it is guaranteed to be **infrequent** and can be pruned immediately without scanning the database.

```text
       {A, B, C}          <-- If {A, B} is infrequent, {A, B, C}
      /    |    \             is immediately pruned without counting!
  {A, B} {A, C} {B, C}
    | \   / \   / |
   {A}   {B}   {C}
```

---

### 2.2 Execution Mechanics: Two-Phase Workflow

Apriori divides the problem into two distinct phases:

```text
+-------------------------------------------------------------+
|                PHASE 1: Frequent Itemset Mining             |
|                                                             |
| Transactions                                                |
|     ↓                                                       |
| Level k=1: Count frequency of single items                  |
|     ↓                                                       |
| Prune 1-itemsets with support < min_support  --> L_1        |
|     ↓                                                       |
| Loop k = 2, 3, ... until L_k is empty:                      |
|     1. Candidate Generation: Join L_{k-1} with itself --> C_k|
|     2. Candidate Pruning: Remove candidates with any         |
|        infrequent (k-1)-subset                              |
|     3. Support Counting: Scan transactions to count         |
|        occurrences of remaining candidates in C_k           |
|     4. Filter: Retain candidates with support >= min_support|
|        --> L_k                                              |
+-------------------------------------------------------------+
                              ↓
+-------------------------------------------------------------+
|                 PHASE 2: Rule Generation                    |
|                                                             |
| For each frequent itemset f in L_k (for k >= 2):            |
|     1. Generate all non-empty proper subsets A ⊂ f          |
|     2. Form rule: A --> (f \ A)                             |
|     3. Compute confidence = support(f) / support(A)         |
|     4. Filter rules where confidence >= min_confidence      |
+-------------------------------------------------------------+
```

---

## 3. Under the Hood in PHP-ML (`Phpml\Association\Apriori`)

PHP-ML implements Apriori in `Phpml\Association\Apriori`, utilizing the `Trainable` and `Predictable` traits.

### Class Architecture & Execution Pipeline

```text
   +-------------------------------------------------------------+
   |                     Phpml\Association\Apriori               |
   +-------------------------------------------------------------+
   | - $support: float                                           |
   | - $confidence: float                                        |
   | - $samples: array                                           |
   | - $large: array           // Frequent itemsets L[k]         |
   | - $rules: array           // Generated association rules    |
   +-------------------------------------------------------------+
          |
          +--> train($samples, $targets = [])
          |      Stores transactions in $this->samples.
          |
          +--> getRules()
          |      Triggers apriori() and generateAllRules() lazily.
          |
          +--> apriori()
          |      Calculates L[1] via $this->items() and $this->frequent().
          |      Iteratively builds L[k] using $this->candidates().
          |
          +--> generateAllRules()
          |      Explores power sets for each frequent itemset.
          |
          +--> predict(array $samples)
                 Matches input samples against rule antecedents.
```

### Key Internal Methods

1. **`items()`:**
   Scans all transaction samples, extracts unique items, and builds candidate 1-itemsets: `[['milk'], ['bread'], ...]`.
2. **`candidates(array $samples)`:**
   Generates $k$-itemsets from $(k-1)$-itemsets:
    - Pairs $p$ and $q$ from $L_{k-1}$.
    - Tests if they differ by exactly one item each:
        ```php
        if (count(array_merge(array_diff($p, $q), array_diff($q, $p))) != 2) {
            continue;
        }
        ```
    - Combines them into a candidate $k$-itemset: `array_unique(array_merge($p, $q))`.
    - Checks if at least one sample contains the candidate before adding it.
3. **`generateRules(array $frequent)`:**
    - Computes all non-empty proper subsets using `$this->powerSet($frequent)`.
    - For each subset $A$, sets $A$ as `antecedent` and $frequent \setminus A$ as `consequent`.
    - Computes confidence:
      $$\text{confidence} = \frac{\text{support}(frequent)}{\text{support}(antecedent)}$$
    - If $\text{confidence} \ge \text{min\_confidence}$, creates the rule entry:
        ```php
        [
            'antecedent' => $antecedent,
            'consequent' => $consequent,
            'support'    => $this->support($frequent),
            'confidence' => $confidence,
        ]
        ```
4. **`predictSample(array $sample)`:**
    - Filters generated rules where the antecedent matches `$sample`:
        ```php
        $this->equals($rule[self::ARRAY_KEY_ANTECEDENT], $sample)
        ```
    - Returns an array containing the `consequent` of each matching rule.

---

## 4. Parameter Classes & Configuration Scenarios

### 4.1 Constructor Hyperparameters

```php
$apriori = new Apriori(float $support = 0.0, float $confidence = 0.0);
```

| Parameter     | Type    | Default | Domain       | Description & Effect                                                          |
| :------------ | :------ | :------ | :----------- | :---------------------------------------------------------------------------- |
| `$support`    | `float` | `0.0`   | `[0.0, 1.0]` | Minimum relative frequency required for an itemset to be considered frequent. |
| `$confidence` | `float` | `0.0`   | `[0.0, 1.0]` | Minimum conditional probability required for an association rule to be kept.  |

#### Tuning & Trade-offs

- **High Support (e.g., `0.4` - `0.6`):**
    - Keeps only widespread, dominant patterns.
    - Fast execution; low memory footprint.
    - Risk: Misses niche, high-confidence associations.
- **Low Support (e.g., `0.01` - `0.05`):**
    - Captures rare patterns across large datasets.
    - Risk: Combinatorial explosion of candidates and exponential RAM consumption.
- **Default `(0.0, 0.0)`:**
    - On real datasets, retaining all combinations causes severe slowdown or memory exhaustion. **Always specify explicit thresholds.**

---

### 4.2 Training Parameters (`train`)

```php
$apriori->train(array $samples, array $targets = []);
```

- **`$samples` (`array<int, array<int, mixed>>`):**
    - The dataset of transactions. Each row is a transaction containing item identifiers (strings, integers, etc.).
- **`$targets` (`array`):**
    - Unsupervised learning has no target labels.
    - Because `Apriori` implements the `Trainable` trait, the parameter exists in the signature. Pass an empty array: `[]`.

---

### 4.3 Prediction Parameters (`predict`)

```php
$result = $apriori->predict(array $samples);
```

The method accepts a batch array of query itemsets: `array<int, array<int, mixed>>`.

There are three primary input query patterns:

#### Pattern A: Single 1-Item Query

```php
$result = $apriori->predict([['milk']]);
```

- Looks for rules where antecedent is exactly `['milk']`.

#### Pattern B: Multi-Item Query (Compound Antecedent)

```php
$result = $apriori->predict([['milk', 'bread']]);
```

- Looks for rules where the antecedent contains both `'milk'` and `'bread'`.

> [!IMPORTANT]
> **PHP-ML Exact Set Equality Quirk:**
> PHP-ML's `predictSample()` checks rule antecedents using `equals()`:
>
> ```php
> return array_diff($set1, $set2) == array_diff($set2, $set1);
> ```
>
> This requires **exact set equality**, NOT a subset check.
>
> - If you pass `[['milk', 'bread']]`, PHP-ML will **only** return rules whose antecedent is precisely `{'milk', 'bread'}` (such as `{milk, bread} -> {cheese}`).
> - It will **not** trigger `{milk} -> {cheese}`, even though `milk` is present in your input.

#### Pattern C: Batch Query (Multiple Samples at Once)

```php
$result = $apriori->predict([
    ['milk'],
    ['fish'],
    ['bread', 'butter'],
]);
```

- Index `0` corresponds to recommendations for `['milk']`.
- Index `1` corresponds to recommendations for `['fish']`.
- Index `2` corresponds to recommendations for `['bread', 'butter']`.

---

## 5. Handling Responses & Multi-Element Outputs

### 5.1 The 3-Dimensional Response Structure

The return value of `$apriori->predict($samples)` is a **three-level nested array**:

$$\text{Result}[\text{Sample Index}][\text{Rule Index}][\text{Item Index}]$$

```php
$result = $apriori->predict([['milk'], ['fish']]);
```

Produces:

```text
Array
(
    [0] => Array               // Sample 0: ['milk']
        (
            [0] => Array       // Rule 1 consequent (1 item)
                (
                    [0] => bread
                )
            [1] => Array       // Rule 2 consequent (1 item)
                (
                    [0] => cheese
                )
            [2] => Array       // Rule 3 consequent (MULTIPLE ITEMS!)
                (
                    [0] => bread
                    [1] => cheese
                )
        )
    [1] => Array               // Sample 1: ['fish']
        (
                               // Empty array: no matching rules found
        )
)
```

---

### 5.2 Case 1: Multiple Items in a Single Rule Consequent

Association rules are **not restricted to single-item outcomes** ($1 \to 1$). The consequent $Y$ can be an itemset of cardinality $\ge 2$:

$$X \implies \{Y_1, Y_2, \dots, Y_m\}$$

**Why does this happen?**
When the algorithm discovers a frequent $k$-itemset where $k \ge 3$ (e.g., $\{\text{milk}, \text{bread}, \text{cheese}\}$), Phase 2 generates every proper non-empty subset as an antecedent.

When the antecedent is 1 item ($\{\text{milk}\}$), the consequent is the remaining set ($\{\text{bread}, \text{cheese}\}$):

$$\{\text{milk}\} \implies \{\text{bread}, \text{cheese}\}$$

If this rule satisfies minimum confidence, PHP-ML returns:

```php
['bread', 'cheese']
```

as a single rule's consequent.

---

### 5.3 Case 2: Multiple Rules Matching One Query Sample

A single query antecedent may satisfy multiple distinct rules simultaneously. For example:

1. $\{\text{milk}\} \implies \{\text{bread}\}$ (confidence = $80\%$)
2. $\{\text{milk}\} \implies \{\text{cheese}\}$ (confidence = $75\%$)
3. $\{\text{milk}\} \implies \{\text{bread}, \text{cheese}\}$ (confidence = $60\%$)

If your minimum confidence threshold is $50\%$, **all three rules are valid**.
Therefore, querying for `['milk']` produces an array containing all three consequent lists:

```php
$result[0] = [
    ['bread'],
    ['cheese'],
    ['bread', 'cheese'],
];
```

---

### 5.4 Best Practices for Unpacking and Flattening Results

#### Handling a Single Basket Query

```php
$result = $apriori->predict([['milk']]);

// $result[0] contains all matching rule consequents for the first basket:
$rulesConsequents = $result[0];

// Flatten all items across all matched rules:
$allItems = array_merge(...$rulesConsequents);

// Deduplicate items to form a clean recommendation list:
$uniqueRecommendations = array_values(array_unique($allItems));

// Output: ['bread', 'cheese']
```

#### Handling Batch Predictions

When predicting for multiple baskets at once, **do not** use `array_merge(...$result)` indiscriminately, as it merges recommendations across completely different query samples.

```php
$queries = [
    ['milk'],
    ['fish'],
    ['eggs', 'bacon'],
];

$results = $apriori->predict($queries);

foreach ($results as $index => $consequents) {
    $basket = $queries[$index];
    $basketLabel = implode(', ', $basket);

    if (empty($consequents)) {
        echo "Basket [{$basketLabel}]: No recommendations found.\n";
        continue;
    }

    // Flatten and deduplicate per basket
    $flatItems = array_values(array_unique(array_merge(...$consequents)));
    echo "Basket [{$basketLabel}] recommends: " . implode(', ', $flatItems) . "\n";
}
```

---

## 6. End-to-End Practical Example

Below is a complete, executable demonstration of Apriori illustrating compound antecedents, multi-element consequents, and batch prediction:

```php
<?php

require_once __DIR__ . '/../../vendor/autoload.php';

use Phpml\Association\Apriori;

// 1. Transaction dataset
$samples = [
    ['milk', 'bread', 'cheese'],
    ['milk', 'bread', 'cheese'],
    ['milk', 'bread'],
    ['milk', 'cheese'],
    ['bread', 'cheese'],
    ['fish', 'shrimp', 'salmon'],
];

// 2. Instantiate with 40% support and 50% confidence
$support = 0.4;
$confidence = 0.5;
$apriori = new Apriori($support, $confidence);

// 3. Train (targets is empty array)
$apriori->train($samples, []);

// 4. Inspect generated association rules
echo "=== GENERATED RULES ===\n";
foreach ($apriori->getRules() as $rule) {
    $ant = implode(', ', $rule['antecedent']);
    $con = implode(', ', $rule['consequent']);
    $sup = round($rule['support'] * 100, 1);
    $cnf = round($rule['confidence'] * 100, 1);
    echo "Rule: [{$ant}] => [{$con}] (Support: {$sup}%, Confidence: {$cnf}%)\n";
}

// 5. Predict with diverse queries
echo "\n=== PREDICTIONS ===\n";
$queries = [
    ['milk'],                  // Single item antecedent
    ['milk', 'bread'],         // Multi-item antecedent
    ['fish'],                  // Antecedent from separate domain
    ['unknown_item'],          // Unmatched item
];

$predictions = $apriori->predict($queries);

foreach ($queries as $i => $query) {
    $qStr = implode(', ', $query);
    $consequents = $predictions[$i];

    if (empty($consequents)) {
        echo "Query [{$qStr}] -> (No rules matched)\n";
        continue;
    }

    $uniqueSuggestions = array_values(array_unique(array_merge(...$consequents)));
    echo "Query [{$qStr}] -> [" . implode(', ', $uniqueSuggestions) . "]\n";
}
```

---

## 7. Summary & Quick Reference

| Feature                   | Behavior in PHP-ML `Apriori`                                                                     |
| :------------------------ | :----------------------------------------------------------------------------------------------- |
| **Learning Paradigm**     | Unsupervised rule association.                                                                   |
| **Target Variable**       | Pass `[]` into `$apriori->train($samples, [])`.                                                  |
| **Prediction Matching**   | **Exact set equality** on antecedents (`equals()`). Does not perform partial/subset matching.    |
| **Response Format**       | 3D Array: `[sample_idx][rule_idx][item_idx]`.                                                    |
| **Multi-Item Consequent** | Yes, occurs when rule consequent contains $>1$ items derived from frequent $(k \ge 3)$-itemsets. |
| **Multi-Rule Matches**    | Yes, one antecedent can match multiple rules with different consequents.                         |
| **Default Thresholds**    | `0.0` support, `0.0` confidence (always set explicit values to avoid exponential blowup).        |
