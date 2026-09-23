from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score,f1_score,roc_curve, roc_auc_score,ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd



# open datfile
with open("data/dialog_acts.dat","r") as file:
    data = file.readlines()

#TODO use groupfold sklearn to make each duplicate group a group and make sure when splitting they are either all in the test set or in the training set



# parse data so y and x are separated
def parse_data(data):
    x_data = []
    y_data = []
    for i in range(len(data)):
        temp = data[i].split(" ", 1)
        x_data.append(temp[1].lower())
        y_data.append(temp[0].lower())

    return x_data,y_data
        
x,y = parse_data(data)


#split the data in 85/15 split
def data_split_random(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.15,random_state=15,stratify=y)
    return x_train, x_test, y_train, y_test
    



# turn data into a Bow representa
vectorizer = CountVectorizer()
bow_vector = vectorizer.fit_transform(x)


# initiliaze training and test set
x_train, x_test, y_train, y_test = data_split_random(bow_vector.toarray(),y)




# initilize logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)


y_pred = model.predict(x_test)
np.set_printoptions(linewidth=np.inf, threshold=np.inf)
print(y_pred)
print(y_test)

labels = list(dict.fromkeys(y_test))
print(labels)

cm = confusion_matrix(y_test,y_pred)
recall = recall_score(y_test,y_pred,average='micro',labels=np.unique(y_pred))
accuracy = accuracy_score(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

print(cm)
print(accuracy)
print(recall)

