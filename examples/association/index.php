<?php

require_once __DIR__ . '/../../vendor/autoload.php';

// Phpml provides Association Rule Mining via the Apriori algorithm.
// Unlike SVM (classification), Apriori is an UNSUPERVISED learning algorithm
// used for "Market Basket Analysis" to uncover relationships between items.
use Phpml\Association\Apriori;

/*
 * 1. INSTANTIATING THE ALGORITHM & CORE METRICS
 *
 * The constructor accepts two critical threshold parameters:
 *   Apriori(float $support = 0.1, float $confidence = 0.4)
 *
 * - Support: How frequently an itemset appears across all transactions.
 *     Support(X) = (Transactions containing X) / (Total transactions)
 *     Low support weeds out rare, insignificant combinations.
 *
 * - Confidence: How likely item Y is purchased when item X is purchased.
 *     Confidence(X -> Y) = Support(X and Y) / Support(X)
 *     Measures the reliability of the inferred rule.
 *
 * Default values used here: support = 0.1 (10%), confidence = 0.4 (40%).
 */
$apriori = new Apriori();

/*
 * 2. TRANSACTION DATABASE (ITEMSETS)
 *
 * In association mining, data consists of discrete baskets (transactions),
 * not numerical coordinates.
 *
 * Total transactions (N) = 4:
 *   Basket 1: ['milk', 'bread', 'cheese']
 *   Basket 2: ['fish', 'shrimp', 'salmon']
 *   Basket 3: ['eggs', 'bacon', 'toast']
 *   Basket 4: ['sun', 'clouds', 'rain']
 *
 * Note: Each basket here contains completely disjoint items (no item overlaps
 * across baskets), so each individual item appears exactly 1 out of 4 times (Support = 25%).
 */
$samples = [
    ['milk', 'bread', 'cheese'],
    ['fish', 'shrimp', 'salmon'],
    ['eggs', 'bacon', 'toast'],
    ['sun', 'clouds', 'rain'],
];

/*
 * 3. MODEL TRAINING & RULE EXTRACTION
 *
 * - Unsupervised Nature: Notice the second argument is an empty array `[]`.
 *   There are no target labels (no ground-truth output classes like in SVM).
 *
 * What happens internally:
 * 1. Frequent Itemset Generation: Finds all combinations of items that meet the
 *    minimum Support threshold (0.1, or 10%). Since all items appear 25% of the time,
 *    combinations within the same basket are identified as frequent itemsets.
 * 2. Rule Derivation: From frequent itemsets, it calculates directional implications
 *    (e.g., {milk} -> {bread, cheese}) and retains only those meeting minimum Confidence (40%).
 */
$apriori->train($samples, []);

/*
 * 4. INFERENCE / RECOMMENDATIONS
 *
 * Querying with antecedent items to retrieve consequent items based on mined rules:
 * - Query 1: ['milk'] -> Given 'milk' in the basket, what associated items follow?
 *   Rule: {milk} -> {bread}, {cheese}, or {bread, cheese} with 100% confidence.
 * - Query 2: ['fish'] -> Given 'fish' in the basket, what associated items follow?
 *   Rule: {fish} -> {shrimp}, {salmon}, or {shrimp, salmon} with 100% confidence.
 */
$result = $apriori->predict([['milk'], ['fish']]);

/*
 * 5. FLATTENING AND DISPLAYING ASSOCIATIONS
 *
 * $result contains an array of recommended itemsets for each input query.
 * array_merge(...$result) flattens them for unified iteration.
 */
$associations = array_merge(...$result);
foreach ($associations as $association) {
    // Prints the recommended consequent items (e.g., "bread, cheese", "shrimp, salmon")
    echo implode(', ', $association) . PHP_EOL;
}
