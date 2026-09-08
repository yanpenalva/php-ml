# Apriori

Apriori finds frequent itemsets in transaction data and can use them to generate association rules.

## Core concepts

- **Transaction:** one observed basket or event, containing a set of items.
- **Item:** one value in a transaction, such as `bread`.
- **Itemset:** a set of one or more items, such as `{milk, bread}`.
- **Candidate itemset:** an itemset being evaluated for frequency.
- **Frequent itemset:** an itemset whose support reaches the minimum-support threshold.
- **Association rule:** an implication `X -> Y`, where `X` and `Y` do not overlap.

Support measures how often an itemset occurs:

```text
support(X) = transactions containing X / total transactions
```

Confidence measures how often `Y` appears when `X` appears:

```text
confidence(X -> Y) = support(X ∪ Y) / support(X)
```

Rules are kept when their confidence reaches the minimum-confidence threshold.

## Algorithmic mechanics

Apriori relies on the anti-monotonic property:

```text
X ⊆ Y => support(Y) <= support(X)
```

Adding items cannot make an itemset occur in more transactions. Therefore, if an itemset is infrequent, every larger itemset containing it is also infrequent. Apriori uses this fact to avoid counting many impossible candidates.

For each level `k`, it joins compatible frequent `(k-1)`-itemsets to generate candidate `k`-itemsets. It then prunes candidates whose `(k-1)`-subsets are not all frequent, counts support in the transactions, and removes candidates below minimum support. The remaining itemsets become the input for the next level.

For each frequent itemset, rule generation tests non-empty subsets as possible antecedents. The remaining items form the consequent, and confidence filters the resulting rules.

```text
transactions
    ↓
generate candidate itemsets
    ↓
count support
    ↓
remove candidates below minimum support
    ↓
generate larger candidates
    ↓
repeat
    ↓
generate association rules
    ↓
filter by confidence
```

The search space can grow exponentially with the number of distinct items: there can be up to `2^n - 1` non-empty itemsets. At a high level, each iteration costs work proportional to the number of candidates, the number of transactions, and the cost of testing item containment. Join and prune reduce this practical cost, but the algorithm can still become expensive when support thresholds are low or the item vocabulary is large.
