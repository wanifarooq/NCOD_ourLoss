# Improving Learning in the Presence of Noisy Labels by Combining Distance-to-Class Centroids and Outlier Discounting

[![Paper](https://img.shields.io/badge/paper-arXiv%3A2007.00151-green)](https://arxiv.org/abs/2202.14026)

</div>
This repository serves as a practical implementation of our novel loss function, NCOD (Improving Learning in the Presence of Noisy Labels by Combining Distance-to-Class Centroids and Outlier Discounting), following the structural framework of the SOP paper. We have deliberately introduced synthetic noise, as outlined in the literature when dealing with noisy label experiments. Most of the code has been adapted from previous repositories contributed by individuals addressing the same problem. However, we've made specific modifications to accommodate our unique loss function. Additionally, the repository includes code files for 3D embedding representation and other experimental details.

### Example
Please follow Table 4 for hyperparameters and change the hyperparameters accordingly for example
For 50% symmetric noise to cifar 100 for non ensembled architecture
```
python train.py -c configuration.json --lr_u 0.1  --percent 0.5
```
For 40% Asymetric noise 
```
python train.py -c configuration.json --lr_u 0.3  --percent 0.4  --asym True
```


