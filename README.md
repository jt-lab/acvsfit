# ACVSfit - A framework for fitting Adaptive Choice Visual Search data

*ACVSfit* is a library for fitting data from Adaptive Choice Visual Search (ACVS) experiments and for conducting a full-fledged Bayesian analysis. ACVS experiments (e.g., Irons & Leber, [2016](#irons2016),  Bergmann et al., [2020](#bergmann2020),  Mu et al., [2024](#mu2024)) produce data where the choice between two targets fluctuates back and forth between preference for one or the other. *ACVSfit* implements hierarchical Bayesian models to fit such data and obtain three meaningful parameters: *adtaptation* (how strongly observers adapt to changing distracotr set sizes), *shift* (how much the adaptation lags behind the objective changes in the displays), and *bias* (the strength of the general preference for one of the two target types). *ACVSfit* provides tools for estimating the model, performing prior and posterior predictive checks, and various diagnostics and plots. It aims to provide the tools required to produce analysis reports that follow the BARG suggestions (Kruschke, [2021](#kruschke2021)).

*ACVSfit* is still under heavy development. The current version 0.3.0 has a level of completeness that covers a few concrete ongoing ACVS projects. However, it is intended to extend the framework to a level so that it can be easily used for diverse ACVS experiments. The current version can already be useful in many cases. Restrictions of this version are listed below under "Restrictions of version 0.3.0".

## Versioning

*ACVSfit* uses the following versioning scheme to enable reproducible analysis:
MAJOR.MINOR.UPDATE, the current version is 0.3.0

* MAJOR reflects the level of completeness. When 1 is reached, the framework should generally be applicable to diverse ACVS experiments, and the documentation is complete. 

* MINOR, importantly, indicates versions with which analysis outcomes remain reproducible. That is, within one MINOR version, default priors or other parts of the model will not change. 

* UPDATE versions might include improvements to plots, workflows, fixing of problems (as long as they do not change analysis outcomes), etc.

## Restrictions of version 0.3.0:

* Documentation is still minimalistic. Feel free to get in touch if you need help applying the framework.

* *acvsfit* includes a model for within-participant conditions that uses correlated varying effects, which helps the model to partially pool information across conditions and participants. However, this model is not yet compatible with all diagnostic and plotting functions. Hence, to use this model, it might be required to manually process the traces with the ArviZ library.

* Some (plotting) functions might take the input data frame as an argument, even though they are not plotting any of the raw data. This is because labels, such as names of the target types, are extracted from the input data and used for labeling plots.


## References

<a id ="mu2024"></a>Mu, Y., Schubö, A., & Tünnermann, J. (2024). Adapting attentional control settings in a shape-changing environment. *Attention, Perception, & Psychophysics, 86*(2), 404–421.

<a id ="bergmann2020"></a>Bergmann, N., Tünnermann, J., & Schubö, A. (2020). Which search are you on? Adapting to color while searching for shape. *Attention, Perception, & Psychophysics, 82*(2), 457–477.

<a id="irons2016"></a>Irons, J. L., & Leber, A. B. (2016). Choosing attentional control settings in a dynamically changing environment. *Attention, Perception, & Psychophysics, 78*(7), 2031–2048.

<a id="kruschke2021"></a>Kruschke, J. K. (2021). Bayesian analysis reporting guidelines. *Nature Human Behaviour, 5*(10), 1282–1291.
