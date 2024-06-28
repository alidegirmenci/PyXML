Maximum Absolute Scaling
========================

In the maximum absolute scaling method, the data is scaled so that the highest 
value of the feature is 1. If the data contains negative values, the range of 
the scaled data is limited between -1 and 1. Maximum absolute scaling defined as

.. math:: 
    {x_{maximum\_{\rm{a}}bsolute\_scaled}} = \frac{x}{{\max \left( {\left| x \right|} \right)}}

* :math:`x` is the feature value
* :math:`{\max \left( {\left| x \right|} \right)}` is the maximum absolute value of the feature


The result of applying the maximum absolute scaling method to the uploaded 
dataset is shown in Figure 20. Maximum absolute scaled train and test sets 
can be viewed by clicking "Show Training Data" to view the train data and 
"Show Test Data" to view the test data.

.. _fig20:

.. figure:: images/figure_20.png
   :alt: Application of maximum absolute scaling
   :align: center

   **Figure 20:** Application of maximum absolute scaling