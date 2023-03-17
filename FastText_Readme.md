## Why use Fast text? 
Efficient learning of word representation and text classification

## Perks: 
- Trains models faster than other libraries

## Fast Text Functionality:

 Train Model ->  Predict classifications -> Evaluate model

- General text classification methods require high computational power because the model computes scores for each and every label along with the correct label in the training set.

- Fast text uses Hierarchical Classifier ( A binary Tree), Instead of computing score for every label fast text just computes probability of each node on the path of the correct label which decreases the number of computations for each piece of text.

- Note: Uses softmax to calculate the probability 

- Calculates the probability for every label and outputs the highest probability label to classify the input text.