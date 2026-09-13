# Regression Analysis in PHP-ML

Regression is a supervised machine-learning task used to predict a continuous
numeric value. Instead of selecting a discrete class, a regression algorithm
learns a relationship between input features and a target such as price,
temperature, or sales.

## 1. Core Concepts

- **Feature vector (`x`)**: numeric inputs describing one observation.
- **Target (`y`)**: the continuous value to predict.
- **Training set**: paired samples and targets, passed to `train()`.
- **Prediction (`ŷ`)**: the numeric value produced for a new sample.
- **Residual**: the prediction error, `y - ŷ`.

Regression differs from classification because its output is a number, not a
label such as `A` or `B`.

## 2. Least Squares Linear Regression

PHP-ML's `LeastSquares` learns a linear relationship. With one feature, the
model is:

```text
ŷ = b₀ + b₁x
```

With multiple features, it becomes:

```text
ŷ = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
```

The algorithm chooses the intercept `b₀` and coefficients `bᵢ` that minimize
the sum of squared residuals:

```text
SSE = Σ(yᵢ - ŷᵢ)²
```

Squaring makes positive and negative errors comparable and penalizes large
errors more strongly. In matrix form, PHP-ML uses the normal equation:

```text
b = (XᵀX)⁻¹Xᵀy
```

The implementation adds a column of `1`s to `X` so the first coefficient is
the intercept. `train()` computes and stores the coefficients; `predict()`
then evaluates the linear equation for each new feature vector.

## 3. Support Vector Regression (SVR)

`SVR` adapts the support-vector idea to numeric targets. It attempts to fit a
function inside an epsilon tube around the observed values. Errors inside the
tube are ignored, while errors outside it are penalized. Kernels such as RBF
can model non-linear relationships, and the `epsilon` and `cost` parameters
control tolerance and error penalties.

Unlike `LeastSquares`, SVR uses the external libsvm implementation bundled by
PHP-ML, so it is useful for more complex or non-linear data.

## 4. PHP-ML API

```php
use Phpml\Regression\LeastSquares;

$regressor = new LeastSquares();
$regressor->train($samples, $targets);
$predictions = $regressor->predict($testSamples);
```

`$samples` is a two-dimensional feature matrix and `$targets` is a one-
dimensional numeric array with one target per sample. The example in
`examples/regression/index.php` uses a small linear dataset so the learned
coefficients and predictions are easy to inspect.

## 5. Evaluation Metrics

- **MAE**: `mean(|y - ŷ|)`; average absolute error in the target's units.
- **MSE**: `mean((y - ŷ)²)`; emphasizes larger errors.
- **RMSE**: `sqrt(MSE)`; returns to the target's units.
- **R²**: `1 - SSE/SST`; indicates how much target variance the model explains.

Metrics should be calculated on unseen test data. A low training error alone
does not prove that the model generalizes.

## 6. Practical Considerations

- Use numeric features and keep feature dimensions consistent.
- Linear regression works best when the relationship is approximately linear.
- Strongly correlated features can make coefficients unstable.
- Outliers can strongly affect least-squares coefficients because errors are
  squared.
- Split data into training and test sets for a realistic evaluation.
