# Literature review

## Controlled leaf recognition

Mohanty, Hughes and Salathé [1] established deep-learning plant-disease classification on PlantVillage and discussed evaluation beyond controlled imagery. High within-dataset accuracy alone therefore cannot establish deployment readiness. Our study measures separate source and external performance without borrowing their reported accuracy as a baseline result. [Publisher](https://doi.org/10.3389/fpls.2016.01419).

## External data and domain shift

Singh et al. [2] introduced PlantDoc to broaden visual conditions beyond laboratory-style images. The supplied classification repository contains cropped images; this is distinct from the detection benchmark. Dataset collection and annotation affect what external validity can be claimed. Our class exclusions are more conservative than equating every similar-looking label. [Paper](https://arxiv.org/abs/1911.10317).

Wu et al. [5] investigate unsupervised domain adaptation for laboratory-to-field plant recognition. Their setting uses target-domain information for adaptation. Our protocol holds PlantDoc out of fitting altogether, so results are not directly comparable to an adaptation method. This distinction prevents conflating domain generalization evaluation with target-domain adaptation. [Publisher](https://doi.org/10.34133/plantphenomics.0038).

## Calibration

Guo et al. [3] examine confidence calibration in modern neural networks and scalar temperature scaling. A positive temperature can change confidence without changing the predicted class. Their work motivates measuring reliability separately from accuracy; it does not establish that source-calibrated probabilities remain calibrated under agricultural domain shift. [Publisher](https://proceedings.mlr.press/v70/guo17a.html).

## Selective prediction

Geifman and El-Yaniv [4] formalize rejection-based classification with risk/coverage analysis. Our lightweight validation-threshold rule is an empirical comparison baseline; it does not implement their probabilistic risk bounds. This project reports realized external risk instead of claiming that a validation target guarantees safety. [Publisher](https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html).

## Closely overlapping prior work and defensible contribution

Xiang, Shi and Zhu [6] directly study reliability under PlantVillage-to-PlantDoc shift, including calibration and selective prediction. Consequently, neither the dataset pairing nor a calibration-and-abstention analysis is a new topic. Our contribution is an independently reproducible, resource-aware ResNet-18/MobileNetV3-Small comparison with explicit mapping/exclusions, saved provenance, source-only selection and a dashboard that serves real artifacts. This is an educational empirical contribution, not an asserted methodological novelty or publication claim. [Publisher](https://doi.org/10.3389/fpls.2026.1826962).

## Gaps this implementation can investigate

The actionable question is whether the two selected architectures differ in the joint trade-off among external accuracy, calibration, abstention coverage and local inference cost under one fixed protocol. Any answer remains pending full runs. Additional seeds, stronger near-duplicate auditing, independent calibration partitions and prospectively collected field images would make follow-up conclusions more credible. Reference numbers correspond to [references.md](references.md).
