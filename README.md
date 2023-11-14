# EX 9-Implementation-of-SVM-For-Spam-Mail-Detection
## DATE:09.11.23
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Collect a labeled dataset of emails, distinguishing between spam and non-spam.

2.Preprocess the email data by removing unnecessary characters, converting to lowercase, removing stop words, and performing stemming or lemmatization.

3.Extract features from the preprocessed text using techniques like Bag-of-Words or TF-IDF.

4.Split the dataset into a training set and a test set.

5.Train an SVM model using the training set, selecting the appropriate kernel function and hyperparameters.

6.Evaluate the trained model using the test set, considering metrics such as accuracy, precision, recall, and F1 score.

7.Optimize the model's performance by tuning its hyperparameters through techniques like grid search or random search.

8.Deploy the trained and fine-tuned model for real-world use, integrating it into an email server or application.

9.Monitor the model's performance and periodically update it with new data or adjust hyperparameters as needed

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: DIVYA.A
RegisterNumber: 212222230034
*/
import chardet
file='/content/spam.csv'
with open(file, 'rb') as rawdata:
  result = chardet.detect(rawdata.read(1000000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,
random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

### RESULT OUTPUT:

![1](https://github.com/Divya110205/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119404855/e033efb7-1223-46ea-89c2-a94121f64c43)

### data.head():

![2](https://github.com/Divya110205/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119404855/31860742-08dc-4381-a085-43be1ab6a2c5)

### data.info():

![3](https://github.com/Divya110205/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119404855/23ede932-69cd-4b5f-92d2-ad554b3aece2)

### data.isnull().sum():

![4](https://github.com/Divya110205/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119404855/dfb29571-12e1-47b4-aaed-5b2411f10113)

### Y_prediction VALUE:

![5](https://github.com/Divya110205/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119404855/3e2a6c84-5c11-4a83-a640-39813e664070)

### ACCURACY VALUE:

![6](https://github.com/Divya110205/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119404855/7198fc82-cc4d-4bd5-8f8d-2fa6497727ed)

## Result:

Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

