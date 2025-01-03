# CineReviewX - Movies Sentiment Analyzers

## Table of Contents
- [About The Project](#about-the-project)
  - [Key Steps Involved](#key-steps-involved)
  - [Key Features](#key-features)
- [About the Data](#about-the-data)
  - [Target Variable](#target-variable)
  - [Dataset Source Link](#dataset-source-link)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)


## About The Project

**CineReviewX** is an IMDB movie sentiment analysis project that classifies movie reviews into two categories: positive or negative. Using deep learning techniques, this project processes textual movie reviews to predict the sentiment behind them. By leveraging Recurrent Neural Network (RNN) models, it analyzes large datasets of reviews, providing insights into how audiences emotionally react to films.

### Key Steps Involved
1. **Data Collection:** The project uses a dataset of movie reviews from IMDB, containing both positive and negative feedback from users.

2. **Data Preprocessing:** The raw text data is tokenized and converted into vectors using One-Hot Encoding, transforming the text into a numerical format suitable for analysis.

3. **Model Building:**  The model is constructed using the following layers:
   - **Embedding Layer:** Converts words into dense vector representations.
   - **SimpleRNN Layers:** Captures patterns in the sequence of words to understand sentiment.
   - **Dense Output Layer:** Uses a sigmoid activation function to output a probability value, classifying the sentiment as either positive (1) or negative (0).

4. **Model Training:**   The model is trained on the preprocessed dataset, utilizing appropriate optimization techniques to minimize loss and improve accuracy. The early stopping mechanism monitors validation loss and halts training when performance plateaus.

5. **Model Evaluation:** After training, the model is evaluated based on metrics such as accuracy, precision, recall, and F1-score to assess its performance on unseen data.

6. **Deployment:** A **Streamlit** web application is developed to allow users to input movie reviews and get real-time sentiment predictions.

### Key Features
- **Sentiment Classification**: Classifies movie reviews as Positive or Negative.
- **Data Preprocessing**: Prepares and cleans raw text for machine learning models.
- **Text Vectorization**: Converts raw text into numerical format using One-Hot Encoding.
- **Modeling**: Implements a Simple RNN for sentiment analysis.
- **Evaluation Metrics**: Measures model performance using accuracy.

![cineReviewX](cineReviewX.png)

 ## About the Data

The dataset contains a collection of IMDB movie reviews and their corresponding sentiment labels (positive or negative). It includes user reviews, movie ratings, and associated metadata for each review. The dataset is preprocessed to remove irrelevant information such as stop words, HTML tags, and special characters.

## Target Variable

The target variable for this project is **Sentiment** (either 1 for positive or 0 for negative), which is derived from the movie reviews. The model's goal is to predict the sentiment label based on the review text.

## Dataset Source Link

The dataset used in this project is publicly available on [Kaggle IMDB Dataset](https://www.kaggle.com/). You can download it directly from there for local use or to train the models.

## Technology Stack

- Python
- Streamlit
- TensorFlow
- Tensorboard
- Scikit-learn
- Keras
- Pandas
- NumPy
- Pickle

## Getting Started

To get started with this project locally, you’ll need Python 3.10+ installed on your machine along with some necessary Python packages. You can either clone the repository and install dependencies manually or use Docker for an isolated environment.

### Installation Steps

1. Clone the repository:

   - Open your terminal or command prompt.
   - Navigate to the directory where you want to install the project.
   - Run the following command to clone the GitHub repository:
     ```
     git clone https://github.com/shubhamprajapati7748/CineReviewX.git
     ```

2. Create a Virtual Environment (Optional)

   - It's a good practice to create a virtual environment to manage project dependencies. Run the following command:
     ```
     conda create -p <Environment_Name> python==<python version> -y
     ```

3. Activate the Virtual Environment (Optional)

   - Activate the virtual environment based on your operating system:
     ```
     conda activate <Environment_Name>/
     ```

4. Install Dependencies

   - Navigate to the project directory:
     ```
     cd [project_directory]
     ```
   - Run the following command to install project dependencies:
     ```
     pip install -r requirements.txt
     ```

5. Run the Project
    ```bash
    streamlit run app.py
    ```

6. Access the Project
   - Visit `http://localhost:8501` in your browser to use the app.


## Contributing

We welcome contributions to improve this project! Whether you are fixing bugs, adding features, or improving documentation, feel free to fork the repository and submit a pull request.

### Steps to contribute:
1. Fork the repo.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add feature'`).
5. Push to your branch (`git push origin feature-name`).
6. Create a new Pull Request.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Shubham Prajapati - [@shubhamprajapati7748@gmail.com](shubhamprajapati7748@gmail.com)

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/): For providing the machine learning framework to train the predictive model.
- [Streamlit](https://streamlit.io/): For creating the interactive web application.
- [Scikit-learn](https://scikit-learn.org/): For preprocessing utilities such as scaling and encoding.
- [Kaggle](https://www.kaggle.com/): For the inspiration behind the dataset, which is similar to the dataset used in this project.

