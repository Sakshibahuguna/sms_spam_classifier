# Importing necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset
messages = pd.read_csv("spam.csv", encoding='latin-1')

# Rename columns
messages = messages.rename(columns={'v1': 'label', 'v2': 'message'})

# Keep only required columns
messages = messages[['label', 'message']]

# Data preprocessing
lemmatizer = WordNetLemmatizer()
corpus = []

for i in range(len(messages)):
    review = messages['message'][i]
    review = review.lower()
    review = review.split()

    review = [
        lemmatizer.lemmatize(word)
        for word in review
        if word not in set(stopwords.words('english'))
    ]

    review = ' '.join(review)
    corpus.append(review)

# Bag of Words model
cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(corpus).toarray()

# Encode labels
y = messages['label'].map({'ham': 0, 'spam': 1}).values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

print("\nAccuracy:", accuracy)

# Test with a custom message
msg = input("\nEnter your message: ")

sample_message = cv.transform([msg]).toarray()

prediction = model.predict(sample_message)

if prediction[0] == 1:
    print("\nPrediction: SPAM")
else:
    print("\nPrediction: HAM")