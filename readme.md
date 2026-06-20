# SMS Spam Classifier

This Python script analyzes SMS messages to classify them as **spam** or **ham (non-spam)**. It utilizes the **Multinomial Naive Bayes algorithm** and Natural Language Processing (NLP) techniques for text preprocessing and classification.

## Usage

### Dependencies

Ensure you have Python installed on your system. You'll also need to install the following libraries:

* pandas
* nltk
* scikit-learn

You can install them using pip:

```bash
pip install pandas nltk scikit-learn
```

### Download NLTK Data

NLTK requires additional data to be downloaded. Run the following commands in a Python shell or script:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### Run the Script

```bash
python3 sms_classifier.py
```

## File Description

* **sms_classifier.py** : Python script for SMS spam classification.
* **spam.csv** : Dataset containing SMS messages and their labels.
* **README.md** : This file providing instructions and information about the project.

## Dataset

The script expects the dataset to be in CSV format with two columns representing the message labels and SMS text. The dataset used for this project is the SMS Spam Collection Dataset.

* **ham** : Normal messages
* **spam** : Unwanted or promotional messages

You can download the dataset from Kaggle:

https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

## Model

The project uses:

* CountVectorizer (Bag of Words)
* Multinomial Naive Bayes Classifier

## Performance

The model achieves an accuracy of approximately **98.74%** on the test dataset.

## Acknowledgments

This script was created as part of a machine learning project. The dataset used for training and testing the model is the SMS Spam Collection Dataset available on Kaggle.

 
