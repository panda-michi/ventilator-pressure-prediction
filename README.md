# ventilator-pressure-prediction
Code for the top 4%(silver medal) of https://www.kaggle.com/c/ventilator-pressure-prediction

## environment
- Google Colab Pro(2021-11-05)
- python: 3.7.12
- tensorflow: 2.7.0
- numpy: 1.19.5
- pandas: 1.3.4
- pytorch: 1.10.0+cu111
- pytorch-lightning: 1.5.2
- transformers: 4.12.5

## features
- [toda's features](https://www.kaggle.com/takamichitoda/ventilator-train-classification): /presprocess/create_toda_features.py
- [dlaststark's features](https://www.kaggle.com/dlaststark/gb-vpp-pulp-fiction): add_features_v7 at /preprocess/add_features.py

## training
- model: LSTM based on https://www.kaggle.com/dlaststark/gb-vpp-pulp-fiction
- pretraining: learning with mae loss
	- 15 folds cross validation with dlaststark's features
	- 10 folds cross validation with toda's features
- finetuning: learning with masked mae loss
	- exclude u_out=1