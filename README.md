<h1 align="center">Welcome to bookie_xgb_analysis 👋</h1>
<p>
</p>

Bookie football predictions overview:
<ol type="1">
  <li>https://github.com/nichohelmut/bookie_weekly_import_ms</li>
  <li>https://github.com/nichohelmut/bookie_clustering_ms</li>
  <li>https://github.com/nichohelmut/bookie_xgb_preprocess</li>
  <li><b>https://github.com/nichohelmut/bookie_xgb_analysis</b></li>
  <li>https://github.com/nichohelmut/bookie_result_check</li>
</ol>

This microservice is run a GCP Cloud Engine VM. The VM is triggered every Thursday by a Pub/Sub event and saves the results in Bigquery Table.
<p>
</p>
In this microservice we will then fit and train a XGBoost model with the pre-developed and self-engineered features from the previous <a href="https://github.com/nichohelmut/bookie_clustering_ms">clustering</a> and <a href="https://github.com/nichohelmut/bookie_xgb_preprocess">preprocessing</a>preprocessing services.
<p>
</p>
‘XGBoost is an open source library providing a high-performance implementation of gradient boosted decision trees.’ Instead of just building multiple trees at parallel, it builds them sequentially in order to reduce the errors from the previous tree.
<p>
</p>
While creating the XGBoost classifier, as parameter: ‘objective’, we will use ‘multi:softprob’, where the result will contain predicted probability of each data point belonging to each class. This is helpful since we do not have a binary, but a multi classification problem.

## Author

👤 **Nicholas Utikal**

* Website: https://medium.com/@nicholasutikal/predicting-football-results-using-archetype-analysis-and-xgboost-1344027eae28
* Github: [@nichohelmut](https://github.com/nichohelmut)
* LinkedIn: [@https:\/\/www.linkedin.com\/in\/nicholas-utikal\/](https://linkedin.com/in/https:\/\/www.linkedin.com\/in\/nicholas-utikal\/)

## Show your support

Give a ⭐️ if this project helped you!
