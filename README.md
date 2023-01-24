# Machine-Learning-Audio-Classification
![image](https://user-images.githubusercontent.com/98383338/213648125-a0fcc678-06bc-4e18-b1ea-d8ec01b22e92.png) ![image](https://user-images.githubusercontent.com/98383338/213648223-88ce3cd7-be18-43e7-ab62-e97f3dc81504.png)

### Classifying Indoor and Outdoor acoustic environment

Using the MLEnd London Sounds dataset, building a machine learning pipeline that takes as an input an audio segment and predicting whether the audio segment has been recorded indoors or outdoors. The problem here is to classify the 7 seconds audio by intrepreting its acoustic environment Indoor or Outdoor.

We will be using Supervised Machine Learning Models by using features from the librosa python package to retrive information.

*What's Intresting about it*

Audio files are complex data types. Specifically they are discrete signals or time series, consisting of values on a temporal window/dimension.This project gives an insight to work with different audio features to classify its environment.

Additionally this project gives an inference of machine learning model that best suits this application. Furthermore, We will explore different methods to choose best hyperparameteters and compares the effectiveness of supervised models for this application.

Machine Learning Pipeline

Input

The Input for the ML pipeline consists of a csv file and collection of 7 seconds audio clips(.wav files) recorded in different filiming locations of London.

The csv file consists of name of the audio files along with the Area Spot, environment, Participant Number recorded in different filiming locations of London.

Feature Extraction

In this stage we are loading a collection of audio files which are imported from google drive along with the dataframe where name of the audio file is given as index. Here we are extracting spectral features like    using librosa library using different functions. It returns and array(X) of these features and a binary boolean label y which is True for Indoor and False for Outdoor.

Feature Analysis

In this stage data having high skewness is transformed into low skewness using logorithmic, square, cube root math functions. We are reducing the skewness so that the data has smoothned curve.If the skewness of the data lies between -1 to 1 it is normally distributed. Data is inconsistent so the data is standardised by using standard scaler from sklearn library where the transformed data has mean as zero and varience as 1.

Training and Tuning the model

The processed data is given as input to the models after spliting into 70% Training and 30% Testing data. Followed by the Supervised Machine Learning Models to predict labels as Indoor and Outdoors. Each model is accesed for evaluation using accuracy metric, precision, recall, f1-score. Using RandomizedSearchCV best hyperparameters are selected that best suits the model.

Model Evaluation

In this stage we will further select the best model using accuracy obtained for each model on the validation dataset(30%). The final model is selected based on accuracy, confusion matrix, classification report so that the suitable model is not biased towards any predictors. For each model we are predicting output based on the indoor or outdoor environment
