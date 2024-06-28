Median and Quantile Scaling
===========================

Median and quantile scaling, also known as robust scaling, uses statistics 
(which are the median and quantiles of the feature) to scale data. In this 
type of scaling, the median value is subtracted from the value of the 
feature and scaled by the interquartile range. Median and quantile scaling 
defined as

.. math:: 
    {x_{median\_quantile\_scaled}} = \frac{{x - median(x)}}{{IQR\left( x \right)}}

* :math:`x` is the feature value
* :math:`median\left( x \right)` median(x) is the median value of the feature 
* :math:`IQR\left( x \right)` is the interquartile range of the feature 


The result of applying the median and quantile scaling method to the uploaded 
dataset is shown in Figure 21. Median and quantile scaled train and test sets 
can be viewed by clicking "Show Training Data" to view the train data and 
"Show Test Data" to view the test data.

.. _fig21:

.. figure:: images/figure_21.png
   :alt: Application of median and quantile scaling
   :align: center

   **Figure 21:** Application of median and quantile scaling