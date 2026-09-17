<?php

require_once __DIR__ . '/../../../vendor/autoload.php';

// Phpml implements KMeans as an unsupervised clustering algorithm. Unlike
// the classifiers seen so far, it receives no labels: it discovers groups
// in the data by itself, which is why this folder is called "grouping".
use Phpml\Clustering\KMeans;

/*
 * 1. INSTANTIATING THE CLUSTERER
 *
 * The constructor argument is k: how many groups we want the algorithm to
 * find. Here k = 2. Choosing k is the user's responsibility — the algorithm
 * never decides how many groups exist, it only partitions the data into
 * exactly k of them.
 */
$clustering = new KMeans(3);

/*
 * 2. UNLABELED SAMPLES (INPUTS)
 *
 * Each row is one point [x, y]. Notice there is NO target array and NO
 * label array: in clustering, the "answer" is the group itself. The two
 * natural clouds here are around x = 1 and x = 4, and the algorithm should
 * detect that structure on its own.
 */
$samples =
[
[1,2],
[1,4],
[1,0],
[4,2],
[4,4],
[4,0],
[4,1],
[21,31],
[22,32],
[23,33],
[24,34],
[25,35]
];

/*
 * 3. CLUSTERING
 *
 * This single call replaces the train()/predict() pair used by supervised
 * algorithms. KMeans:
 *   1. Places k=2 centroids (moving "group centers") in feature space.
 *   2. Assigns every sample to its nearest centroid.
 *   3. Moves each centroid to the mean of its assigned samples.
 *   4. Repeats steps 2-3 until assignments stop changing.
 *
 * It returns an array of k arrays: $result[0] and $result[1] hold the
 * samples belonging to group 1 and group 2. Samples inside a group are
 * close to each other; samples in different groups are far apart.
 */
$result = $clustering->cluster($samples);

/*
 * 4. BUILDING READABLE OUTPUT (FIRST FOREACH)
 *
 * This loop walks over the groups, not over the samples. Each iteration
 * handles ONE whole cluster:
 *   - $cluster is an array containing every sample assigned to group $i.
 *   - 'group1:' / 'group2:' is created as a string header for that group.
 * The nested loop then appends this group's points to the same string,
 * producing one output line per group. $i only tracks which group number
 * we are formatting; it is not an index into $samples.
 */
$grouped = [];
$i = 0;
foreach ($result as $cluster) {
    $grouped[$i] = 'group ' . ($i + 1) . ':';

    /*
     * 5. WALKING ONE GROUP (SECOND, NESTED FOREACH)
     *
     * This inner loop iterates over the samples OF THE CURRENT GROUP.
     * $sample is a point like [1,4] — KMeans returns the raw samples
     * themselves, not indexes into $samples. We append it formatted as
     * "[1,4]," to the group's string. When the inner loop ends,
     * $grouped[$i] holds the complete line for this group, and the
     * outer loop moves to the next cluster.
     */
    foreach ($cluster as $sample) {
        $grouped[$i] .= "[" . $sample[0] . "," . $sample[1] . "],";
    }
    $i++;
}

/*
 * 6. OUTPUT RESULTS
 *
 * The last foreach does no math at all: $grouped is already a list of
 * ready-made strings, one per group, and we simply print one line each.
 */
foreach ($grouped as $values) {
    echo $values . PHP_EOL;
}
