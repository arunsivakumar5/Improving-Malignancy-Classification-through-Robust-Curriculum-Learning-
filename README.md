# Improving-Malignancy-Classification-through-Robust-Curriculum-Learning
Curriculum task learning in combination with worst group generalization leads to improved performance in atypical subtypes from malignant category

# Introduction
Deep learning models used in Computer-Aided Diagnosis (CAD) systems are often trained with Empirical Risk Minimization (ERM) loss. These models often achieve high overall classification accuracy but with lower classification accuracy on certain subgroups. In the context of lung nodule malignancy classification task, these atypical subgroups exist due to the lung
cancer heterogeneity. In this study, we characterize lung nodule malignancy subgroups using the malignancy likelihood ratings given by radiologists and improve the worst subgroup performance by utilizing group Distributionally Robust Optimization (gDRO). However, we noticed that gDRO improves on worst subgroup performance from the benign category, which has less
clinical importance than improving classification accuracy for a malignant subgroup. Therefore, we propose a novel curriculum gDRO training scheme that trains for an “easy” task (nodule
malignancy is determinate or indeterminate for radiologists) first, then for a “hard” task (malignant, benign, or indeterminate nodule). Our results indicate that our approach boosts the worst group subclass accuracy from the malignant category, by up to 6 percentage points compared to standard methods that address and improve worst group classification performance.
# Dataset:
The LIDC dataset [32,33] contains 2,680 distinct nodules in computed tomography (CT) scans from 1,010 patients; nodules of three millimetres or larger are manually identified, delineated, and semantically characterized by up to four different radiologists across nine semantic characteristics (features).

# Algorithm: Curriculum gDRO
Input: Training data D1 with two superclass labels, training
data D2 with three superclass labels, and validation data
Dval with three superclass labels.

Stage one: Training a gDRO model on the easier task
for certain number of epochs, this is an additional
hyperparameter which is tuned for our method. We name
it as threshold epoch e
Train model fgDRO on D1 (Binary classification) via
gDRO.

Stage two: Training gDRO model on the harder task after
threshold epoch
Train model f gDRO on D2 (three-class classification)
via gDRO and then selecting the best model in terms of
worst-group accuracy on Dv al as the final model.

## References:
[1]Miller, K. D., Nogueira, L., Devasia, T., Mariotto, A. B., Yabroff, K.
R., Jemal, A., Kramer, J., Siegel, R. L. (2022). Cancer treatment
and survivorship statistics, 2022. CA: A Cancer Journal for Clinicians.
https://doi.org/10.3322/caac.21731

[2] G. K. Abraham, P. Bhaskaran, and V. S. Jayanthi, “Lung nodule clas-
sification in CT images using Convolutional Neural Network,” 2019 9th
International Conference on Advances in Computing and Communication
(ICACC), 2019.

[3] Oakden-Rayner, L., Dunnmon, J., Carneiro, G., and R ́e, C., “Hidden
stratification causes clinically meaningful failures in machine learning
for medical imaging,” in [Proceedings of the ACM conference on health,
inference, and learning], 151–159 (2020).

[4] Mastouri, R.; Khlifa, N.; Neji, H.; Hantous-Zannad, S. A bilinear con-
volutional neural network for lung nodules classification on CT images.
Int. J. Comput. Assist. Radiol. Surg. 2021, 16, 91–101

[5] Yu, J., Yang, B., Wang, J., Leader, J. K., Wilson, D. O., and Pu, J., “2d
cnn versus 3d cnn for false-positive reduction in lung cancer screening,”
Journal of Medical Imaging 7(5), 051202 (2020).

[6] Zhao, X., Qi, S., Zhang, B., Ma, H., Qian, W., Yao, Y., and Sun, J., “Deep
cnn models for pulmonary nodule classification: model modification,
model integration, and transfer learning,” Journal of X-ray Science and
Technology 27(4), 615–629 (2019)

[7] A. Nibali, Z. He, and D. Wollersheim, “Pulmonary nodule classification
with deep residual networks,” International Journal of Computer Assisted
Radiology and Surgery, vol. 12, no. 10, pp. 1799–1808, 2017.

[8] T. Zeng, E. Furst, Y. Wang, R. Tchoua, J. D. Furst, and D. S. Raicu, ”No
Nodule Left Behind: Evaluating Lung Nodule Malignancy Classification
with Different Stratification Schemes,” presented at the SPIE Medical
Imaging Symposium, San Diego, CA, USA, Feb. 19-23, 2023

[9] J. Wen et al., ”Subtyping brain diseases from imaging data,” arXiv
preprint arXiv:2202.10945, 2022.

[10] A. Sotiras, S. M. Resnick, and C. Davatzikos, ”Finding imaging patterns
of structural covariance via non-negative matrix factorization,” Neuroim-
age, vol. 108, pp. 1-16, 2015.

[11] J. Wen et al., ”Multi-scale semi-supervised clustering of brain images:
Deriving disease subtypes,” Medical Image Analysis, vol. 75, p. 102304,
2022.

[12] A. Ezzati, A. R. Zammit, C. Habeck, C. B. Hall, and R. B. Lipton,
”Detecting biological heterogeneity patterns in ADNI amnestic mild
cognitive impairment based on volumetric MRI,” Brain Imaging and
Behavior, vol. 14, no. 5, pp. 1792-1804, 2020.

[13] J. Chen, L. Milot, H. Cheung, and A. L. Martel, ”Unsupervised
clustering of quantitative imaging phenotypes using autoencoder and
gaussian mixture model,” in International Conference on Medical Image
Computing and Computer-Assisted Intervention, 2019: Springer, pp. 575-
582.

[14] Z. Yang, J. Wen, and C. Davatzikos, ”Surreal-GAN: Semi-Supervised
Representation Learning via GAN for uncovering heterogeneous disease-
related imaging patterns,” presented at the International Conference on
Learning Representations (ICLR), Virtual, 2022.

[15] Z. Shen et al., ”Towards out-of-distribution generalization: A survey,”
arXiv preprint arXiv:2108.13624, 2021.

[16] W. Hu, G. Niu, I. Sato, and M. Sugiyama, ”Does distributionally robust
supervised learning give robust classifiers?,” in International Conference
on Machine Learning, 2018: PMLR, pp. 2029-2037.

[17] M. Basseville, ”Divergence measures for statistical data processing—An
annotated bibliography,” Signal Processing, vol. 93, no. 4, pp. 621-633,
2013.

[18] L. R ̈uschendorf, ”The Wasserstein distance and approximation theo-
rems,” Probability Theory and Related Fields, vol. 70, no. 1, pp. 117-129,
1985.

[19] H. Namkoong and J. C. Duchi, ”Stochastic gradient methods for distri-
butionally robust optimization with f-divergences,” Advances in Neural
Information Processing Systems, vol. 29, 2016.

[20] T. Hashimoto, M. Srivastava, H. Namkoong, and P. Liang, ”Fairness
without demographics in repeated loss minimization,” in International
Conference on Machine Learning, 2018: PMLR, pp. 1929-1938.

[21] A. Sinha, H. Namkoong, and J. Duchi, ”Certifying Some Distributional
Robustness with Principled Adversarial Training,” in International Conference on Learning Representations, 2018.

[22] Sagawa, S., Koh, P. W., Hashimoto, T. B., and Liang,P. Distributionally
robust neural networks for group shifts: On the importance of regularization for worst-case generalization. In International Conference on
Learning Representations (ICLR), 2020a.

[23] J. Byrd and Z. Lipton, ”What is the effect of importance weighting in
deep learning?,” in International Conference on Machine Learning, 2019:
PMLR, pp. 872-881.

[24] Sohoni, N. S., Dunnmon, J. A., Angus, G., Gu, A., and R ́e, C. No sub-
class left behind: Fine-grained robustness in coarse-grained classification
problems. arXiv preprint arXiv:2011.12945, 2020.

[25] H. Ye, C. Xie, Y. Liu, and Z. Li, ”Out-of-distribution generalization
analysis via influence function,” arXiv preprint arXiv:2101.08521, 2021.

[26] T. Nguyen, Z. Hongyang, H. Nguyen “Improved Group Robust-
ness via Classifier Retraining on Independent Splits” arXiv preprint
arXiv:2204.09583,2022

[27] Yaghoobzadeh, Y., Mehri, S., Tachet, R., Hazen, T. J., and Sordoni, A.
Increasing robustness to spurious correlations using forgettable examples.
arXiv preprint arXiv:1911.03861, 2019.

[28] E. Xu, T. Ramaraj, R. Tchoua, J. Furst, and D. Raicu, “Contextualizing
lung nodule malignancy predictions with easy vs. Hard Image Classi-
fication,” 2022 Fourth International Conference on Transdisciplinary AI
(TransAI), 2022.

[29] J. Wei, A. Suriawinata, B. Ren, X. Liu, M. Lisovsky, L. Vaickus, C.
Brown, M. Baker, M. Nasir-Moin, N. Tomita, L. Torresani, J. Wei, and S.
Hassanpour, “Learn like a pathologist: Curriculum learning by annotator
agreement for Histopathology Image Classification,” 2021 IEEE Winter
Conference on Applications of Computer Vision (WACV), 2021.

[30] A. Jim ́enez-S ́anchez, D. Mateus, S. Kirchhoff, C. Kirchhoff, P. Bib-
erthaler, N. Navab, M. Gonz ́alez Ballester, and G. Piella, “Medical-based
deep curriculum learning for improved fracture classification,” Lecture
Notes in Computer Science, pp. 694–702, 2019.

[31] J. Luo, D. Arefan, M. Zuley, J. H. Sumkin, and S. Wu, “Deep curriculum
learning in task space for multi-class based mammography diagnosis,”
Medical Imaging 2022: Computer-Aided Diagnosis, 2022.

[32] S. G. Armato III et al., “Lung image database consortium: developing
a resource for the medical imaging research community,” Radiology, vol.
232, no. 3, pp. 739-748, 2004.

[33] M. F. McNitt-Gray et al., “The Lung Image Database Consortium
(LIDC) data collection process for nodule detection and annotation,”
Academic radiology, vol. 14, no. 12, pp. 1464-1474, 2007 Magnetics
Japan, p. 301, 1982].
