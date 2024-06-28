SHAP 
====

SHAP is based on the concept of Shapley value and coalitional game theory 
and was introduced by Lundberg and Lee [2]_. It allows the machine learning model 
to be explained both globally and locally. After activating the SHAP method 
via the radio button, SHAP hyperparameters can be set on the left side of the 
window, such as the sample number, maximum features for local explanation, and 
maximum features for global explanation. The SHAP method is adequately explained 
in the article for this software, but more information can be found in the 
original article.

The implementation of the SHAP method is slightly different from the 
implementation of other XML (LIME, ANCHORS) methods in PyXML. After selecting 
the SHAP method via the radio buttons at the top of the window, the "Apply SHAP" 
button should be pressed to create the SHAP model, as shown in Figure 46. "Local" 
radio button for local explanations and "Global" radio button for global 
explanations should be selected from the left side of the window. 

.. _fig46:

.. figure:: images/figure_46.png
   :alt: Default explanation window for SHAP method
   :align: center

   **Figure 46:** Default explanation window for SHAP method

When the "Local" radio button is selected, the window in Figure 47 will be 
active, the sample number and maximum feature hyperparameters must be set. 
Local explanations in SHAP allow explanation with four different graphs: which 
are bar plot, decision plot, force plot, and waterfall plot. The user has to 
select one of these four graph types and then click on the "Display SHAP" button 
to visualize the results on the right side of the window. The user can also make 
the explanation in a different plot type by selecting one of the other plot types 
without having to create a SHAP model again. To do this, the user only needs to 
change the active radio button and click on the "Display SHAP" button. The graph 
on the right side of the window will then be automatically replaced with the new 
plot type.

.. _fig47:

.. figure:: images/figure_47.png
   :alt: Local explanation result of the SHAP method with waterfall plot  
   :align: center

   **Figure 47:** Local explanation result of the SHAP method with waterfall plot 

When the "Global" radio button is selected, the window in Figure 48 will be 
active, the maximum feature hyperparameter must be set. Global explanations in 
SHAP allow explanation with six different graphs: which are bar plot, beeswarm 
plot, decision plot, heatmap plot, violin plot, and summary plot. The user has 
to select one of these six graph types and then click on the "Display SHAP" button 
to visualize the results on the right side of the window. The user can also make 
the explanation in a different plot type by selecting one of the other plot types 
without having to create a SHAP model again. To do this, the user only needs to 
change the active radio button and click on the "Display SHAP" button. The graph 
on the right side of the window will then be automatically replaced with the new 
plot type.

.. _fig48:

.. figure:: images/figure_48.png
   :alt: Global explanation result of the SHAP method with summary plot 
   :align: center

   **Figure 48:** Global explanation result of the SHAP method with summary plot 


.. [2] Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in neural information processing systems, 30.