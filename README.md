# Phishing URL Detection using Recurrent Weighted Average

Data Privacy is a serious issue with the advent of Technology and prevalence of Social Networks in today's era. This project aims at tackling one avenue how your private data can be compromised - Phishing websites. 
The `final_dataset.csv` contains ~50,000 URLs with 'Benign' and 'Phishing' classes with equal proportion. The Recurrent Weighted Average architecture is used to train a Deep Neural Network to classify these URLs to their respective classes.
The trained model has achieved an accuracy of 98.6% with the current dataset. 

### Dependencies
* Python 3.0 or higher
* TensorFlow
* Tkinter

### Training the Model
1. First clone the repository and install the above dependencies.

2. To start training the model enter the following command in the terminal:
```
python train.py
```

3. After training in complete, you can test out any URL by running:
```
python use_model.py
```
