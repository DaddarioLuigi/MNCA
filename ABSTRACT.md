# Abstract

Modeling complex biological systems presents significant challenges due to their inherent stochasticity, spatial heterogeneity, and multi-scale dynamics. Traditional Neural Cellular Automata (NCAs) have shown promise in capturing emergent behaviors in biological systems, but their deterministic nature limits their ability to faithfully represent the probabilistic processes underlying biological phenomena such as tissue growth, cell differentiation, and pattern formation.

The Mixture of Neural Cellular Automata (MNCA) model addresses these limitations by integrating probabilistic rule selection and intrinsic noise mechanisms, enabling robust modeling of stochastic biological processes. This thesis presents a comprehensive assessment of MNCA models across diverse experimental settings, systematically evaluating their performance, expressiveness, and applicability to real-world biological systems.

Through extensive experimental analysis, we investigate the behavior of MNCA variants—including standard MixtureNCA and MixtureNCANoise models—under varying hyperparameters, update rules, and dynamic configurations. Our evaluation framework encompasses multiple dimensions: (1) tissue growth simulations, where we assess the model's ability to capture cell division, differentiation, and spatial organization patterns; (2) microscopy image analysis, applying MNCA to real-world biological imaging data; (3) robustness analysis, systematically evaluating model stability under various perturbations and noise conditions; and (4) parameter sensitivity studies, characterizing the impact of architectural choices and hyperparameters on model performance.

The insights gained from these comprehensive evaluations guide the development of potential extensions aimed at improving the model's expressiveness and scalability. By comparing MNCA performance against baseline NCA models and alternative approaches such as agent-based models with ABC parameter inference, we provide a thorough understanding of the model's strengths, limitations, and optimal application domains.

This work contributes to the growing field of computational biology by providing a systematic framework for evaluating and extending neural cellular automata models, with particular emphasis on their application to complex biological systems. The findings from this assessment inform best practices for model selection, parameter tuning, and deployment in biological modeling scenarios, ultimately advancing our ability to simulate and understand stochastic biological processes.


