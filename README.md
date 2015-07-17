Main part of this project is java package info.adamovskiy.nn - set of classes for work with neural networks. This implementation has very bad performance, has no multi-threading (yet). Main goal of it - good maintainability, that allows to experiment with different types of neural networks easily.

Second part is info.adamovskiy.digitrecognizer - javafx-based implementation of digit recognizer via multi-layered preceptron with two possible data sources: generator of little noisy digits or reader of .csv file prepared in advance.

My train.csv file was taken from https://www.kaggle.com/c/digit-recognizer/data.
