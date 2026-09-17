# Clustering (Grouping) Analysis in PHP-ML

Clustering is an **unsupervised** machine-learning task. Unlike classification
or regression, the algorithm receives **no labels and no targets**: it receives
only feature vectors and must discover groups ("clusters") on its own. The
group structure is the answer, not a training signal.

PHP-ML ships three clusterers under `Phpml\Clustering`: `KMeans`, `DBSCAN`,
and `FuzzyCMeans`. This document covers the mathematical foundation common to
all of them, then details each algorithm — with emphasis on K-Means, the one
behind the example in `examples/grouping/kmeans/index.php`.

## 1. Core Concepts & Mathematical Foundation

### 1.1 Key Terminology

- **Cluster**: a group of samples that are close to each other in feature
  space and far from samples of other groups.
- **Centroid (`μₖ`)**: the geometric center (per-dimension mean vector) of
  cluster `k`. It is a computed point, usually not an actual sample.
- **k / c**: the number of clusters. Chosen by the user; the algorithm
  partitions the data into exactly that many groups, never more or fewer.
- **Assignment**: the mapping of every sample to exactly one cluster (hard
  clustering). Fuzzy methods replace it with membership degrees.
- **Euclidean distance** — the default notion of "closeness":

```text
d(x, y) = sqrt( (x₁ - y₁)² + (x₂ - y₂)² + ... + (xₙ - yₙ)² )
```

- **Inertia / within-cluster SSE (`J`)**: the quantity K-Means minimizes:

```text
J = Σₖ Σ_{x ∈ Cₖ} ‖x - μₖ‖²
```

For every cluster, sum the **squared** Euclidean distances between each
member and that cluster's centroid; then sum over all clusters. Low `J` means
tight, compact groups.

### 1.2 Clustering vs Classification

| | Classification | Clustering |
|---|---|---|
| Labels | required (`train(samples, labels)`) | none |
| Call pattern | `train()` then `predict()` | single `cluster()` call |
| Output | one class per sample | k groups of samples |
| Group identity | fixed by the labels | arbitrary (group 1 has no inherent name) |
| Evaluation | accuracy, precision, recall, F1 | geometry only: inertia, silhouette |

Because there are no ground-truth labels, classification metrics do not
apply. Evaluation is covered in Section 5.

---

## 2. Algorithm Guides: What Runs Behind the Scenes

### 2.1 K-Means (Lloyd's Algorithm)

#### The Objective

K-Means searches for the partition of the samples into k clusters `C₁…Cₖ`
that minimizes the inertia `J`. Exhaustive search over all partitions is
combinatorial, so the algorithm uses an iterative heuristic that is
guaranteed to converge — but only to a **local** minimum.

#### The Iteration (Assign → Update)

1. **Initialize** k centroids (strategy matters — see 2.2).
2. **Assignment step**: assign every sample to its nearest centroid. With
   centroids fixed, this is the best possible assignment, so `J` cannot
   increase.
3. **Update step**: move each centroid to the mean of its assigned samples
   (`μₖ = mean of Cₖ` per dimension). For fixed assignments, the mean is the
   point that minimizes the sum of squared distances, so again `J` cannot
   increase.
4. **Repeat** steps 2–3 until no sample changes cluster.

Each step never increases `J`, and `J` is bounded below by 0 — so the loop
must terminate, usually in a handful of iterations. This is coordinate
descent on `J`: the assignment step optimizes memberships, the update step
optimizes centroids.

#### Numeric Walkthrough (the 6-point example)

Input — two visible clouds (the x=1 column and the x=4 column), `k = 2`:

```text
[1,2] [1,4] [1,0] [4,2] [4,4] [4,0]
```

Suppose initialization picked seeds `μ₁ = [1,2]` and `μ₂ = [4,0]`.

**Iteration 1 — assignment** (squared distances shown):

| Sample | d² to μ₁=[1,2] | d² to μ₂=[4,0] | Assigned to |
|---|---|---|---|
| [1,2] | 0 | 16 | C₁ |
| [1,4] | 4 | 25 | C₁ |
| [1,0] | 4 | 9  | C₁ |
| [4,4] | 13 | 16 | C₁ ← the tie-break point |
| [4,2] | 9 | 4 | C₂ |
| [4,0] | 16 | 0 | C₂ |

`[4,4]` is nearly equidistant (13 vs 16); small differences in the seeds
decide it. Result: `C₁ = {[1,2],[1,4],[1,0],[4,4]}`, `C₂ = {[4,2],[4,0]}` —
the 4/2 partition the example actually produces.

**Iteration 1 — update** (per-dimension means):

```text
μ₁ = ((1+1+1+4)/4, (2+4+0+4)/4) = (1.75, 2.5)
μ₂ = ((4+4)/2, (2+0)/2)         = (4, 1)
```

**Iteration 2 — assignment**: `[4,4]` now has d² = 7.31 to μ₁ vs 9.00 to μ₂ —
it stays in C₁. No sample moves → **converged**. Final inertia:

```text
J = 0.81 + 2.81 + 0.81 + 7.31 + 5.31 + 5.31 = 22.375
```

#### Why 4/2 Instead of the "Ideal" 3/3

The intuitive partition — the x=1 column vs the x=4 column, 3/3 with
centroids `(1,2)` and `(4,2)` — has inertia `J = 16`, which is **lower** than
the 22.375 the algorithm converged to. Two runs of the example landed in
local optima (22.375 and 17.5) and neither found the global optimum of 16.

Lessons:

- Cluster sizes are **not** balanced: `new KMeans(2)` promises two groups,
  never "two equal halves". Nothing in `J` rewards balance, only compactness.
- The iteration only *descends* `J`; where it stops depends entirely on the
  seeds. Ties and near-ties (`[4,4]`) make the outcome seed-sensitive.
- Mitigation: run several times and keep the partition with the lowest `J`
  (Section 7).

#### Row Order Still Does Not Matter

As with least-squares regression, every step aggregates over whole clusters
(distance to a centroid, per-dimension mean of members). Shuffling the input
rows does not change the mathematics. What *does* vary is the randomized
initialization — never the row sequence itself.

#### Complexity

For `n` samples, `k` clusters, `d` features and `i` iterations:

```text
O(n · k · d · i)
```

Distance computations dominate. K-Means scales well, which is one reason it
remains the default first clustering algorithm.

### 2.2 Initialization Strategies in PHP-ML

The constructor's second argument selects the seeding method:

```php
public const INIT_RANDOM = 1;
public const INIT_KMEANS_PLUS_PLUS = 2;   // default
```

- **`INIT_RANDOM`**: PHP-ML picks **synthetic** seeds — one random integer
  per dimension inside the bounding box of the data (`Space::getBoundaries()`
  + `random_int()`), not actual samples. Cheap, but seeds may fall in empty
  regions.
- **`INIT_KMEANS_PLUS_PLUS`**: spreads the seeds through the data. The
  textbook weights each point by its *squared* distance to the nearest
  already-chosen seed (`P(x) ∝ D(x)²`); PHP-ML samples by plain distance `D`
  (`random_int(0, (int) $sum)` over accumulated distances in
  `Space::initializeKMPPClusters()`), and its very first seed is simply the
  **first sample inserted** into the `Space` (`SplObjectStorage::current()`),
  not a uniform random pick. Consequences:
  - still nondeterministic — repeated runs can converge differently;
  - the first seed is deterministic given input order, so input order does
    subtly influence seeding in PHP-ML (unlike the textbook version).

### 2.3 DBSCAN (Density-Based)

Groups points that are within `epsilon` of at least `minSamples` neighbors,
then expands each core point's region recursively (`expandCluster()`).
Properties:

- **Infers the number of clusters by itself** — no k parameter.
- Explicitly labels isolated points as **noise** (K-Means must absorb
  outliers into some cluster).
- Finds arbitrary-shaped clusters; K-Means assumes compact, roundish ones.
- Struggles when cluster densities vary: a single global `epsilon` cannot
  fit both dense and sparse regions.

### 2.4 FuzzyCMeans (Soft Clustering)

An extension of K-Means where every sample gets a **membership degree**
between 0 and 1 for every cluster instead of a hard assignment. Membership
is weighted by distance and exponentiated with the fuzziness parameter
(musty `m > 1`, default `2.0`); the update and membership steps alternate
until the membership matrix changes by less than `epsilon`. Useful when
group boundaries genuinely overlap (e.g. a customer that is half "budget"
and half "premium").

---

## 3. Under the Hood in PHP-ML

All clusterers implement the `Phpml\Clustering\Clusterer` interface:

```php
namespace Phpml\Clustering;

interface Clusterer
{
    public function cluster(array $samples): array;
}
```

One call, no `train()`/`predict()` split, and no persisted model: clustering
is a one-shot transform of the whole dataset.

### 3.1 Class Architecture Overview (K-Means)

```text
            +-------------------------------------+
            |      Phpml\Clustering\KMeans        |  public facade
            |  (clustersNumber, initialization)   |
            +-------------------------------------+
                              | builds
                              v
            +-------------------------------------+
            |   Phpml\Clustering\KMeans\Space     |  extends SplObjectStorage
            |  owns points, runs Lloyd iteration  |
            +-------------------------------------+
                     | contains            | produces
                     v                     v
        +----------------------+   +--------------------------+
        |   KMeans\Point       |   |      KMeans\Cluster      |
        | coordinates + label  |<--| extends Point (centroid) |
        | ArrayAccess, Countable|  | + set of member points   |
        +----------------------+   +--------------------------+
```

### 3.2 `KMeans::cluster()` Pipeline

1. `new Space(count(reset($samples)))` — the **first row** defines the
   dimensionality of the space.
2. Each sample is attached as a `Point`, carrying its **original array key
   as label** (`$space->addPoint($sample, $key)`). This is why the returned
   groups keep the original indexes: membership is traceable back to the
   input rows.
3. `initializeClusters(k, initMethod)` builds the seeds, then
   `$clusters[0]->attachAll($this)` — **every point starts in cluster 0**.
4. `do { } while (!$this->iterate($clusters));` — at least one full
   assign/update pass always runs.

### 3.3 `Space::iterate()` Internals

The heart of the Lloyd iteration:

```php
foreach ($clusters as $cluster) {
    foreach ($cluster as $point) {
        $closest = $point->getClosest($clusters);
        if ($closest !== $cluster) {
            $attach[$closest]->attach($point);
            $detach[$cluster]->attach($point);
            $convergence = false;
        }
    }
}
// apply all attach/detach batches, then:
foreach ($clusters as $cluster) {
    $cluster->updateCentroid();
}
return $convergence;   // true only when no point moved
```

- Moves are **batched**: all reassignments of a pass are collected first,
  applied second — no point is evaluated against a half-updated state.
- `updateCentroid()` skips empty clusters (`if ($count === 0) return;`), so
  an emptied cluster freezes its old centroid instead of dividing by zero.
  With unlucky k-means++ seeding, centroids can even collide and a group can
  end up empty — running the example repeatedly, you may see all six points
  land in one group.

### 3.4 `Point` Internals

- `getDistanceWith($point, $precise = true)` accumulates the **squared**
  distance and takes the square root only when `$precise` is true.
- `getClosest()` exploits that optimization: comparing squared distances
  preserves the arg-min, so candidate centroids are compared without any
  square root at all.
- Tie-breaking is a **strict `<`**: the first centroid in iteration order
  wins an exact tie. Combined with `SplObjectStorage` ordering, this makes
  tie outcomes stable within a run but seed-dependent across runs.

### 3.5 `Cluster` Internals

`Cluster extends Point` — a cluster literally *is* a moving centroid that
also carries a set of member points:

```php
public function updateCentroid(): void
{
    $count = count($this->points);
    if ($count === 0) {
        return;   // freeze centroid of an empty cluster
    }
    // centroid[n] = sum of members' coordinate n / count
}
```

The inertia `J` from Section 1.1 is exactly what this code computes and
minimizes: mean per dimension = least-squares-optimal center for fixed
members (the same least-squares idea as linear regression, applied to
grouping).

---

## 4. Parameter Classes & Configuration Scenarios

### 4.1 Input Data Format

```php
$clusterer->cluster(array $samples): array;
```

- **Structure:** `array<int, array<int, float|int>>` — a plain 2-D numeric
  matrix, one row per sample. There is **no** targets array.

> [!CAUTION]
> **Dimensionality comes from the first row.** `Space` is constructed with
> `count(reset($samples))`; a ragged later row throws
> `LogicException: (x,y) is not a point of this space`. Keep every row the
> same length.

### 4.2 Constructor Hyperparameters (KMeans)

```php
new KMeans(int $clustersNumber, int $initialization = KMeans::INIT_KMEANS_PLUS_PLUS);
```

| Parameter | Role | Trade-off |
|---|---|---|
| `$clustersNumber` | k, the exact number of groups produced | Too low merges real groups; too high splits them. Choose via elbow (Section 5), never by the algorithm. |
| `$initialization` | seeding strategy | `INIT_KMEANS_PLUS_PLUS` usually converges to better optima; both remain random, so multiple restarts still pay off. |

The algorithm **never infers k**. If you do not know how many groups exist,
that is DBSCAN's job (Section 2.3).

### 4.3 Reading `cluster()` Output

Returns an array of exactly k arrays:

- `$result[0] … $result[k-1]` — group order is arbitrary and can change
  between runs; group numbers carry no meaning.
- Each inner array holds the raw sample rows, **keyed by their original
  input index** (the label attached in 3.2), so `array_keys($result[$g])`
  tells you which input rows landed in group `$g`.
- There is no model to reuse: to place a *future* sample you must compute
  distances to the centroids yourself (`d²(x, μₖ)`, pick the minimum) —
  PHP-ML does not expose the final centroids through the public API.

---

## 5. Evaluation Without Labels

- **Inertia / elbow method**: run K-Means for k = 1, 2, 3, … and plot `J`.
  Inertia always decreases with k; the "elbow" — where the curve stops
  dropping sharply — suggests a natural k.
- **Silhouette** (conceptual): for each sample, compare its mean distance to
  its own cluster (cohesion) against the nearest other cluster (separation).
  Values near 1 indicate well-separated clusters, near 0 borderline samples.
- **External validation**: if trusted labels exist for a subset, compare
  clusters to them (e.g. purity). Diagnostic only — the algorithm itself
  never saw the labels.

## 6. Choosing Between the Three Clusterers

| | KMeans | DBSCAN | FuzzyCMeans |
|---|---|---|---|
| Number of groups | user-set `k` | inferred | user-set `c` |
| Assignment | hard | hard + noise points | soft (degrees 0–1) |
| Cluster shape assumed | compact/spherical | any | compact/spherical |
| Outlier handling | absorbs them | flags as noise | spreads membership |
| Key constructor args | `k`, init strategy | `epsilon`, `minSamples`, distance metric | `c`, `fuzziness`, `epsilon`, `maxIterations` |

## 7. Practical Considerations

- **Scale the features first.** Euclidean distance is dominated by the
  feature with the largest range (a "salary in reais" column drowns an
  "age" column). Standardize when units differ.
- **Run K-Means several times.** Initialization is random and convergence is
  only local (the 6-point example converged to `J = 22.375` and `17.5` while
  the global optimum is `16`). Keep the partition with the lowest `J`.
- **k is a decision, not an output.** The elbow method and domain knowledge
  choose k; the algorithm only obeys it.
- **Outliers pull centroids.** The mean is not robust: one extreme value can
  drag a centroid and reassign half a cluster. DBSCAN handles outliers
  better.
- **Assume compact, roundish clusters.** K-Means excels at blobs; elongated,
  ring-shaped, or nested structures need DBSCAN or spectral methods outside
  PHP-ML.
