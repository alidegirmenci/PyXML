Dataset Statistics
==================

When the "Dataset Statistics" button is clicked, statistical information 
about each feature of the dataset is calculated and tabulated. The 
calculated statistical information of the attributes are minimum, maximum, 
average, standard deviation, variance, and range. Minimum is the smallest 
value in an attribute, maximum is the largest value and mean is the sum of 
the attribute values divided by the number of elements in the dataset. The 
range is equal to the difference between the maximum and minimum in each 
feature. Variance measures the spread or scattering of data points relative 
to mean, and standard deviation is square root of the variance. Range, mean, 
standard deviation, and variance are defined as

.. math:: 
    range = maximum\left( x \right)*minimum\left( x \right)

.. math:: 
    mean\left( \mu  \right) = \frac{{\sum\limits_{i = 1}^n {{x_i}} }}{n}

.. math:: 
    variance\left( {{\sigma ^2}} \right) = \frac{{\sum\limits_{i = 1}^n {\left( {{x_i} - \mu } \right)} }}{n}

.. math:: 
    standard{\rm{ }}deviation\left( \sigma  \right) = \sqrt {\frac{{\sum\limits_{i = 1}^n {\left( {{x_i} - \mu } \right)} }}{n}}

* :math:`{x_i}` is the  :math:`{i^{th}}` feature of the sample in the data 
* :math:`n` is the number of features in the data

The tabular view of the calculated statistical features is shown in Figure 23.

.. _fig23:

.. figure:: images/figure_23.png
   :alt: Application of normalizer
   :align: center

   **Figure 23:** Computed statistical properties of the dataset

