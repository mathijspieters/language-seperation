# Unsupervised Language Seperation
Pytorch code for unsupervised language seperation.
The model consists of two parts:
* Language selector, predicting a binary value (using STE) that indicates the language
* Two language models, that learn to model the words selected by the language selector

## Train

Run the following command in order to train the model:

`python3 main.py --epochs 200 --batch_size 64 --language spanish`

The possible languages are `spanish` and `hindi`.
