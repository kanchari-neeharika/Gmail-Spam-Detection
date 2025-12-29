import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df = pd.read_csv("C:\\Users\\nehak\\Downloads\\mail_data.csv", encoding='latin-1')
df['Category_Encoded'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
X = df['Message']
Y = df['Category_Encoded']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)
feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
print(f"Original Training data shape: {X_train.shape}")
print(f"Transformed Training data shape (Features): {X_train_features.shape}")
model = LogisticRegression()
model.fit(X_train_features, Y_train)
train_predictions = model.predict(X_train_features)
train_accuracy = accuracy_score (Y_train, train_predictions)
test_predictions = model.predict(X_test_features)
test_accuracy = accuracy_score(Y_test, test_predictions)
print(f"Accuracy on training data: {train_accuracy:.4f}")
print(f"Accuracy on test data: {test_accuracy:.4f}")
print("\nClassification Report on Test Data:")
print(classification_report (Y_test, test_predictions, target_names=['Ham', 'Spam']))
print("\nConfusion Matrix on Test Data:")
print(confusion_matrix(Y_test, test_predictions))

def classify_mail(message, model, vectorizer, doubtful_threshold=0.2):
    input_list = [message]
    input_features = vectorizer.transform(input_list)
    prob_spam = model.predict_proba(input_features)[0][1]
    lower_bound = 0.5 - (doubtful_threshold / 2)
    upper_bound = 0.5 + (doubtful_threshold / 2)
    if prob_spam >= upper_bound:
        category = 'Spam'
        confidence = prob_spam
    elif prob_spam <= lower_bound:
        category = 'Non-spam (Ham)'
        confidence = 1 - prob_spam
    else:
        category = 'Doubtful'
        confidence = None
    print("\n--- Classification Result ---")
    print(f"Message: {message[:50]}...")
    print(f"Probability of SPAM: {prob_spam:.4f}")
    if category == 'Spam':
        print(f"Prediction: {category} (Confidence: {confidence:.4f})")
    elif category == 'Non-spam (Ham)':
        print(f"Prediction: {category} (Confidence: {confidence:.4f})")
    else:
        print(f"Prediction: {category} (Model is uncertain)")
print("\n--- Test the Custom Classifier ---")
spam_message = "URGENT! You have won a £1000 prize. Call 09051010101 now to claim!"
classify_mail(spam_message, model, feature_extraction, doubtful_threshold=0.2)
ham_message = "Hey, let's meet up tomorrow for lunch. How about 12:30?"
classify_mail(ham_message, model, feature_extraction, doubtful_threshold=0.2)
ambiguous_message = "Call me now or reply to this text. I'll be waiting for you."
classify_mail(ambiguous_message, model, feature_extraction, doubtful_threshold = 0.2)

