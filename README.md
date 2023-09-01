# Quantity Mapping Research

Research Fall 2019

## Packages

- **FuzzyWuzzy**: pip install fuzzywuzzy
- **py_stringmatching**: pip install py_stringmatching
- **Pytorch**: conda install -c pytorch pytorch
- **AllenNLP**: pip install allennlp 

## Files

- [**create_sets.py**](create_sets.py): Takes raw data and converts it into training and testing labeled sets. It creates a full version of testing and training data and a reduced version for local testing (prefix 's_').
- [**logistic_binary_match.py**](logistic_binary_match.py): Each quantity entry is compared to each database option and is classified as match or not. The inputs are encoded using a BOW.
- [**logistic_binary_str_comparison.py**](logistic_binary_str_comparison.py): Each quantity entry is compared to each database option and is classified as match or not. The input for the logistic regression are string matching values.
- [**logistic_multiclass.py**](logistic_multiclass.py): Each database option is a class and every input is classified to one of them.
- [**main.py**](main.py): Retrieves files from Dr. Korpusik desktop using retrieveSSH.py, saves data by meal and concats all data, performs exact string matching from string_matching.py.
- [**model_sum.xlsx**](model_sum.xlsx): Results from all experiments using NN: 
  - ReLu & Loss function
  - Batch size
  - Patience
  - Optimizer
  - 2 Linear & Dot product vs 2 Linear & Bilinear vs 1 Linear
  - RNNs
  - Stacked
- [**Neural_Network_test.py**](Neural_Network_test.py): Image classification NN, used to make sure pytorch was working and to practice making a NN. Not used in research.
- [**NN_conv_allennlp.py**](NN_conv_allennlp.py): Convolusional NN model used to classify the user quantities. Still in development and testing.
- [**NN_match_with_allennlp.py**](NN_match_with_allennlp.py): Simple FNN used to classify the user quantities.
- [**NN_test_allennlp.py**](NN_test_allennlp.py): Text classification NN using allennlp, used to make sure allennlp was working and to practice using the library. Not used in research.
- [**organize_food_units.py**](organize_food_units.py): Organizes food_units.csv by adding column names, sorting, and removing nbd letter prefix
- [**readCSV.py**](readCSV.py): Helper file to read and process CSV files.
- [**retrieveSSH.py**](retrieveSSH.py): Helper file to retrieve data using SSH.
- [**string_matching.py**](string_matching.py): Performs string matching using FuzzyWuzzy (Levenshtein distance).
- [**tmp.xlsx**](tmp.xlsx): Temp results from exepriments.

## NN basic architecture

You can find a diagram of the basic NN architecture in Lucidchart [here](https://lucid.app/lucidchart/a94ef55a-f2e4-4995-9ab8-97450776a3f2/edit?viewport_loc=0%2C95%2C1760%2C753%2C0_0&invitationId=inv_6e42a591-6995-4966-9227-29b35968951c) (Login required).