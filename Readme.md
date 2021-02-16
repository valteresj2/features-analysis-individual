# New approach to analysis of variables of a model call of Features Analysis Individual

## Abstract
This approach aims to generate two results, one is a bivariate analysis that is an analysis of the performance of individual resources for the binary target and the other is an analysis of the adherence between a test base and the variables of the production base. Usually when we select resources in the training process of a machine learning model, we always see the performance and forget to compare it with the basic production resources, this is an important point that generates rework, if any variable does not show adherence and some inference in the baseline test will be invalidated.

For this I developed a function that compares the characteristics between the test and the production, the paragraph below will show which techniques were used.
