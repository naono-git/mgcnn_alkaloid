# mgcnn_alkaloid: Molecular Graph Convolutional Neural Networks for Prediction of Alkaloid Biosynthesis Bathways

## Description
Source code for our paper "Classification of alkaloids into starting substances
on biosynthetic pathway using graph
convolutional neural networks".
====

## Requirements
Python >= 3.6.6
NumPy >= 1.15.4
Pandas >= 0.22.0
TensorFlow 1.6.0
DeepChem 2.1.0

## Files
- mgcnn_alkaloid.py: Training and evaluation of Molecular Graph Convolutional Neural Networks for alkaloid biosynthesis pathway prediction.
- data/alkaloid_data.csv: Molecular data of alkaloids.

## Usage
1. Five fold cross validation
> python alkaloid_data.csv

2. Verify one segment of five fold cross validation
> python alkaloid_data.csv [0-4]

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## Author

Naoaki ONO

[naono-git](https://github.com/naono-git)

mailto:nono@is.naist.jp

## Reference
[Currently under review]
