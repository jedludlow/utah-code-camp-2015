Utah Code Camp 2015
======

IPython notebooks and supporting files for
a presentation on the Google Prediction API.
The primary result in the presentation is a **performance
benchmark of the Google Prediction API against 
`sklearn.RandomForestClassifier`**.
Key files include:

* ml_with_goog_pred.ipynb: The main presentation.
  A brief introduction to machine learning,
  an example of predicting Titanic survival using
  the Google Prediciton API, and then a comparitive
  benchmark between `sklearn.RandomForestClassifier`
  and the Google Prediction API. The benchmark was
  conducted as part of a data science competition to
  classify very short Taylor Swift audio clips as
  either *huge hit* or *not hit*.
* sklearn_fft_ave.ipynb: A notebook that details
  the `RamdomForestClassifier` benchmarking solution,
  including the feature engineering to compute frequency
  spectra.
* google_prediction_fft_ave.ipynb: A notebook that
  details the Google Prediction API benchmark solution.
* googleprediction.py: A simple wrapper around the
  Google Predication API functions to make it look
  more like the `scikit-learn` standard interface 
  to models.
* ipython_fft_example.ipynb: A notebook showing a quick
  example of computing FFTs in scientific Python. The
  emphasis is on preserving the physical significance
  of the resulting frequency spectrum.
