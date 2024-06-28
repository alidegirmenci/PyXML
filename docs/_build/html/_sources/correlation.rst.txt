Correlation
===========

Correlation is a statistical technique that allows analyzing connections 
within a dataset. It measures the relationship between two or more variables 
in a dataset. The strength and direction of the correlation are determined by 
the correlation coefficient. The correlation coefficient takes values between 
+1 and -1. As the value approaches +1 or -1, the correlation increases and is 
called high correlation; as the value approaches 0, the correlation decreases 
and is called low correlation. The sign (+/-) of the correlation coefficient 
indicates positive and negative correlation between features. A commonly used 
correlation method in machine learning is the Pearson correlation coefficient 
and is defined as:

.. math:: 
    r = \frac{{\sum {\left( {{x_i} - \overline x } \right)\left( {{y_i} - \overline y } \right)} }}{{\sqrt {\sum {\left( {{x_i} - \overline x } \right)\left( {{y_i} - \overline y } \right)} } }}


* :math:`{x_i}` is individual data points of variable :math:`X`
* :math:`{y_i}` is individual data points of variable :math:`Y`
* :math:`\overline x` is the means of variables :math:`X`
* :math:`\overline y` is the means of variables :math:`Y`

When clicked the "Correlation" button, correlation between the loaded dataset 
is calculated and visualized, as can be seen in Figure 26. The red/blue color 
of each cell indicates the direction of the correlation (positive/negative), 
and the color saturation indicates the strength of the correlation between 
features. Since some datasets have a large number of features, zoom, pan, 
scroll, and save options have been added to the figure for better analysis. 

.. _fig26:

.. figure:: images/figure_26.png
   :alt: Correlation plot for the correlation coefficient matrix
   :align: center

   **Figure 26:** Correlation plot for the correlation coefficient matrix
