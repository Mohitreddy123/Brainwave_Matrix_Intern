# Fake News Detection System
  ● The Fake News Detection System aims to automatically classify news articles as either real or fake using machine learning techniques. This system leverages 
    Natural Language Processing (NLP) to analyze text data and identify patterns that distinguish fake news from authentic articles. The Fake News DetectionSystem 
    uses a Multinomial Naive Bayes classifier trained on labeled datasets of fake and real news articles. It processes the text data, extracts features, and then 
    classifies the content into either category.
# Getting Started
  ● These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notesonhow 
    to deploy the project on a live system.What things you need to install the software and how to install them:

**1. Python 3.6:-**

   ● This setup requires that your machine has python 3.6 installed on it. 
   ● you can refer to this url https://www.python.org/downloads/ to download python.
  
   ● Once you have python downloaded and installed, you will need to setup PATH variables (if you want to run python program directly, detail instructions are 
    below in how to run software section). 
    
   ● To do that check this: https://www.pythoncentral.io/add-python-to-path-python-is-not-recognized-as-an- internal- or-external-command/.
    
   ● Setting up PATH variable is optional as you can also run program without it and more instruction are given below on this topic.
  
**2. Second and easier option is to download anaconda and use its anaconda prompt to run the commands.** 
    
    ● To install anaconda check this url https://www.anaconda.com/download/

**3. You will also need to download and install below 3 packages after you install either python or anaconda from the steps above**
    • Sklearn (scikit-learn)
    • numpy
    • scipy
    • NLTK
  
**Key Components:**

     ● Text Preprocessing: Converts raw text data into numerical features using techniques like TF-IDF (Term Frequency-Inverse Document Frequency).
     
     ● Model: A Multinomial Naive Bayes classifier is used to build the model that predicts whether a news article is real or fake.
     
     ● valuation: The model is evaluated based on various metrics such as accuracy, precision, recall, and F1-score.

## Key Steps to Build the Fake News Detection System
**1. Data Collection:-**

    ● The first step is to gather a labeled dataset containing news articles that are marked as 
      real or fake.
    ●  For this example, assume we have two CSV files:
      • true.csv: Contains real news articles labeled with 1.
      • fake.csv: Contains fake news articles labeled with 0.
   
**2. Data Preprocessing:-**

     • Text Cleaning: Remove unnecessary characters, symbols, and stop words from the articles.
     • Tokenization: Split the text into individual words or tokens.
     • Lowercasing: Convert all the text to lowercase to maintain consistency.
     • Stop Words Removal: Remove common words like "the", "and", which do not contribute to 
        the classification task.
   
**3. Feature Extraction (TF-IDF):-**

    ● Convert the cleaned text data into numerical features using TF-IDF (Term Frequency-Inverse 
      Document Frequency).
      
    ● TF-IDF helps to give importance to the most relevant words in the article while reducing the 
      weight of common words.
   
**4. Model Building:-**

   • Choose a machine learning algorithm for text classification. A commonly used algorithm for 
     this task is Multinomial Naive Bayes, which works well for text data.
     
   • Train the model using the preprocessed and vectorized data.
   
**5. Model Building:-**
   
     ● Choose a machine learning algorithm for text classification.
     
     ● A commonly used algorithm for this task is Multinomial Naive Bayes, which works well for text data.
     
     ● Train the model using the preprocessed and vectorized data.
   
**6. Model Evaluation:-**
   
     ● Evaluate the model's performance using metrics such as accuracy, precision, recall, and F1- 
       score.
     ● Use a confusion matrix to understand the model's ability to distinguish between real and 
       fake news.

**EXAMPLE:-**  Accuracy: 0.95
                   
                  precision      recall     f1-score    support

           0        0.96          0.94      0.95        500
           1        0.94          0.96      0.95        500
           

      accuracy                              0.95        1000
      macro avg      0.95         0.95      0.95        1000
      weighted avg   0.95         0.95      0.95        1000

**7. Deployment (Optional):-**

   ● Save the trained model as a .pkl file for future predictions.
   
   ● Integrate the model into a web-based application for real-time fake news detection.

### Diagram
    Here's a simple flow diagram to explain how the Fake News Detection system works:

    [Data Collection] --> [Data Preprocessing] --> [Feature Extraction (TF-IDF)] ---- 
                                                                                      >
    Train/Test Split]  -->  [Model Building (Naive Bayes)]  --> [Model Evaluation]  

### Conclusion
     This Fake News Detection System demonstrates the power of machine learning in identifying misleading or false information. By combining text processing and        classification techniques, the system can effectively separate fake news from legitimate articles. Further improvements could include exploring deep learning      models or integrating real-time detection systems.

