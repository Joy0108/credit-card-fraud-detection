# Data

`creditcard.csv` - Credit Card Fraud Detection dataset (ULB Machine Learning Group / Worldline),
transactions made by European cardholders over two days in September 2013.

* 284,807 transactions, 492 frauds (0.172 %)
* `Time` - seconds elapsed since the first transaction in the file
* `V1`-`V28` - principal components of the confidential original features (PCA applied by the data provider)
* `Amount` - transaction amount in EUR
* `Class` - 1 = fraud, 0 = legitimate

The file (~150 MB) is not committed. `python -m fraud_detection eda` downloads it automatically from
`https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv` (a public mirror of the
Kaggle dataset `mlg-ulb/creditcardfraud`, licensed ODbL). You can also download it from Kaggle and
place it here.
