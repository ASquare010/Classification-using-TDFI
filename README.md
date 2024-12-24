# Text Classification with Logistic Regression

This project demonstrates a text classification pipeline using Python. It predicts categories for input text questions based on a pre-trained `Logistic Regression` model and TF-IDF vectorization.


## Installation

### Prerequisites
- Python 3.11.5

### Steps
1. Unzip the source code.
2. Navigate to the project directory in your terminal.
3. Install the required dependencies using:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

1. **Prepare Your Dataset**:
    - Ensure your dataset is in a compatible format (e.g., CSV or JSON).
    - Load and split the data using `pandas` and `train_test_split`.

2. **Train the Model**:
    - Use the provided code to train the `Logistic Regression` model with `TfidfVectorizer`.

3. **Predict Categories**:
    - Transform new text input using the fitted vectorizer and predict categories using the trained model.

4. **Test New Questions**:
    - Modify the `new_questions` list in the script to test predictions on new text data.

---

## Example Code

### Predict Categories
```python
# Test on new questions
new_questions = [
    "What is the thickness of the drywall in the residence?",
    "How many filters does JCI owe in their contract?",
    "Who is responsible for waterproofing?",
    "Where can I find the latest drawings?"
]

# Transform questions
new_questions_tfidf = vectorizer.transform(new_questions)

# Predict categories
predictions = model.predict(new_questions_tfidf)

# Display results
for question, category in zip(new_questions, predictions):
    print(f"Question: {question} -> Predicted Category: {category}")