Normalize
=========

Normalization is a preprocessing method that adjusts individual samples 
(often rows in a dataset) to have a standard norm, usually following the 
L1 or L2 norm. This process is done independently for each sample, ensuring 
that the feature values of each sample align with a unit hypersphere in the 
feature space. The objective of normalization is to standardize all samples 
to a similar scale, highlighting the relative importance of different features 
with respect to each other. Normalizer is defined as 

.. math:: 
    {x_{normalized}} = \frac{x}{{\sqrt {\sum\nolimits_{i = 1}^n {x_i^2} } }}

* :math:`x` is the feature value
* :math:`{x_i}` is the  :math:`{i^{th}}` feature of the sample in the data 
* :math:`n` is the number of features in the data


The result of applying the normalizer method to the uploaded dataset is shown 
in Figure 19. Normalized train and test sets can be viewed by clicking 
"Show Training Data" to view the train data and "Show Test Data" to view the 
test data.

.. _fig19:

.. figure:: images/figure_19.png
   :alt: Application of normalizer
   :align: center

   **Figure 18:** Application of normalizer