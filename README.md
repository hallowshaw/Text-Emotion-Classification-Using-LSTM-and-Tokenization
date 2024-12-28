# Text Emotion Detection

This project focuses on the classification of emotions from text using machine learning and deep learning techniques. The model, tokenizer, and preprocessing steps are provided, enabling users to directly utilize the trained model without retraining.

---

## Overview

The primary objective of this project is to detect emotions such as joy, sadness, anger, and more from textual input. A deep learning approach, specifically an LSTM (Long Short-Term Memory) model, is employed for classification. The project also includes a pretrained model and tokenizer for quick deployment.

**Pretrained Resources:**
- **Model Files**: `emotion_recognizer.h5` and `emotion_recognizer.keras`
- **Tokenizer**: `tokenizer.pkl`
- **Usage Script**: `text_emo_detection.py`

---

## Process

### 1. Dataset
The dataset consists of labeled text data representing various emotions. Preprocessing and feature engineering steps ensure compatibility with the LSTM-based deep learning pipeline. You can access the dataset here: [Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp).

### 2. Libraries Used
**Libraries Used**: numpy, string (default library), nltk, scikit-learn, tensorflow, matplotlib, seaborn; install them using:
```bash
pip install numpy nltk scikit-learn tensorflow matplotlib seaborn
```


---

## Workflow

1. **Text Preprocessing**:
   - Remove punctuation and special characters using the `string` library.
   - Tokenize sentences using `nltk.tokenize.word_tokenize`.
   - Remove stopwords with the help of `nltk.corpus.stopwords`.
   - Convert tokens to lowercase.

2. **Feature Engineering**:
   - Use `Tokenizer` from TensorFlow to convert text into sequences.
   - Pad sequences using `pad_sequences` for uniform input length.

3. **Model Architecture**:
   - Build a Sequential model using TensorFlow's Keras API.
   - Include an `Embedding` layer for word embeddings.
   - Add an `LSTM` layer for capturing temporal relationships.
   - Use `Dense` layers for classification.

4. **Evaluation**:
   - Use `classification_report` and `confusion_matrix` from sklearn to analyze performance.
   - Visualize results using `matplotlib` and `seaborn`.

5. **Usage**:
   - Download `emotion_recognizer.h5`, `emotion_recognizer.keras`, and `tokenizer.pkl`.
   - Use `text_emo_detection.py` to make predictions by loading these resources directly.

---

## Contributions

We welcome contributions to enhance this project. To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature-name"`).
4. Push to the branch (`git push origin feature-name`).
5. Submit a pull request for review.

---

## License

This project is licensed under the [MIT License](./LICENSE). Feel free to use, modify, and distribute the project, provided proper credit is given.

---

## Acknowledgments

We thank:
- The creators of the dataset used in this project for their valuable contribution: [Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp).
- The open-source library developers for providing robust tools to streamline the development process.




