This project implements an end-to-end Spam vs Ham Message Detection system using Machine Learning and Natural Language Processing (NLP). 
The dataset is preprocessed to remove noise, handle duplicates, and structure labels correctly, followed by text preprocessing including stopword removal and stemming. 
Messages are vectorized using TF-IDF, and a Naive Bayes classifier is trained to accurately classify messages as spam or legitimate (ham). 
The trained model is saved and deployed using a Streamlit web application, enabling real-time message classification with graphical visualizations. 
This project demonstrates a complete ML pipeline from data cleaning and model training to deployment with an interactive user interface.

step 1:  git clone <your-repo-link>
         cd spam-ham-detector
step 2:  python -m venv venv
         venv\Scripts\activate
step 3:  pip install -r requirements.txt
step 4: Download the dataset from Kaggle - https://www.kaggle.com/datasets/muhammadahmedansari/ham-spam-messages-dataset
        and name it as spam-ham.csv in the folder data inside the main folder
step 5: clean and preprocess the data - python src/clean_dataset.py
step 6: train the machine learning model - python src/train_model.py
step 7: run the streamlit web application - python -m streamlit run app.py

