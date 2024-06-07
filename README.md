# Codebase for "A Robust Treatment Recommendation System in Tabular Electronic Health Records"


This directory contains implementations of RTRS treatment recommendation system using liver cancer ablation EHR dataset.

Simply run python3 -m main.py

### Code explanation

(1) preprocess/load_data.py
- Load data

(2) preprocess/get_dataset.py
- Data preprocessing

(3) preprocess/missing_values_imputation.py
- Imputate missing values in dataset

(4) model/mviilgan.py
- Define MVIIL-GAN

(5) preprocess/representation_learning.py
- Define MSLR framework to encode variables

(6) model/ae.py
- Define and return an autoencoder model

(7) model/evaluate.py
- Performance of computation in treatment recommendation tasks

(8) model/ActiveLearning.py
- Define MSB framework and model

(9) model/ReinforcementLearning.py
- Define ADRL + LC framework

(10) model/TD3/
- Classic TD3 Model

(11) main.py
- A framework for data loading, model construction, result evaluation

(12) arguments.py
- Parameter settings

Note that hyper-parameters should be optimized for different datasets.


## Main Dependency Library

scikit-learn==0.24.2

torch==1.8.2

torchaudio==0.8.2

torchvision==0.9.2
