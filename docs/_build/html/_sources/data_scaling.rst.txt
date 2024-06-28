Data Scaling
============

Data scaling in machine learning is the process of transforming the 
features of a dataset to a similar scale. Scaling the data helps prevent 
features with larger numeric values from dominating those with smaller 
numeric values. It therefore mitigates potential biases in the machine 
learning algorithm.

In this section of the "Preprocessing" tab, one of the scaling options 
is selected via radio buttons and then the "Show Train Data" or 
"Show Test Data" buttons are used to display the result of the scaled 
data. Default window of the "Preprocessing" tab is shown in Figure 16.


.. _fig16:

.. figure:: images/figure_16.png
   :alt: Data scaling part of the "Preprocessing" tab.
   :align: center

   **Figure 16:** Data scaling part of the "Preprocessing" tab.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Preprocessing

   Standard Scaling <standard_scaling>
   Min/Max Scaling <min_max_scaling>
   Normalize <normalize>
   Maximum Absolute Scaling <maximum_absolute_scaling>
   Median and Quantile Scaling <median_quantile_scaling>