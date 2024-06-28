LIME 
====

Locally Interpretable Model Agnostic Explanations (LIME) is proposed by 
Ribeiro et al. [1]_. After activating the LIME method via the radio button, LIME 
hyperparameters such as sample number, discretizer, kernel width, distance 
metric, number of features, and number of samples can be set on the left side 
of the window. The LIME method is adequately explained in the article for 
this software, but more information can be found in the original article.

Pressing the "Apply LIME" button will apply LIME to the machine learning 
method, as shown in Figure 43, once the process is complete a new window 
will open to save the LIME results. 

.. _fig43:

.. figure:: images/figure_43.png
   :alt: Application of the LIME
   :align: center

   **Figure 43:** Application of the LIME

By default, LIME's explanations are displayed in HTML file format. Thus, 
explanations can be saved to the desired file path in the computer memory 
with the specified name via the pop-up window, as can be seen in Figure 44. 
After clicking on the "Save" button in the pop-up window, the explanations 
are saved. 

.. _fig44:

.. figure:: images/figure_44.png
   :alt: Pop-up window to save LIME's explanations 
   :align: center

   **Figure 44:** Pop-up window to save LIME's explanations 

The computer's default web browser is then opened and the results are shown 
in a new tab (Figure 45).

.. _fig45:

.. figure:: images/figure_45.png
   :alt: Explanations of LIME
   :align: center

   **Figure 45:** Explanations of LIME


.. [1] Ribeiro, M. T., Singh, S., & Guestrin, C. (2016, August). " Why should i trust you?" Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1135-1144).