import numpy as np

from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import svm

from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
iris.feature_names.append('avg_sepal')
iris.feature_names.append('avg_petal')


x = iris.data
y = iris.target

avg_sepal = []
avg_petal = []

for i in x:
    avg_sepal.append((i[0] + i[1])/2)
    avg_petal.append((i[2] + i[3])/2)
    
x = np.c_[x, avg_sepal]
x = np.c_[x, avg_petal]

for i in range(0,6,2):
    plt.scatter(x[:,i], x[:,i+1], c = y)
    temp = iris.feature_names[i]
    plt.xlabel(temp)
    temp = iris.feature_names[i+1]
    plt.ylabel(temp)
    plt.plot()
    plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

#trying it with k-means clustering
CLF_k_means = neighbors.KNeighborsClassifier()
CLF_k_means.fit(x_train[:,4:6], y_train)
predict_k_means = CLF_k_means.predict(x_test[:,4:6])
print("K-means accuracy: ", accuracy_score(y_test, predict_k_means))

'''
#plotting code for prediction vs actual
plt.scatter(x_test[:,4], x_test[:,5], c = predict_k_means, marker = 'd')
plt.scatter(x_test[:,4], x_test[:,5], c = y_test, marker = 'o')
plt.title('prediction  vs actual')
plt.xlabel('avg_sepal')
plt.ylabel('avg_petal')
plt.plot()
plt.show()
'''

#trying with decision tree
CLF_dt = DecisionTreeClassifier()
CLF_dt.fit(x_train[:,0:4], y_train)
predict_dt = CLF_dt.predict(x_test[:,0:4])
print("Decision Tree Accuracy: ", accuracy_score(y_test, predict_dt))

#trying with random forest 
CLF_rf = RandomForestClassifier(n_estimators = 100, max_features = 2)
CLF_rf.fit(x_train[:,4:6], y_train)
predict_rf = CLF_rf.predict(x_test[:,4:6])
print("Random Forest Accuracy: ",accuracy_score(y_test, predict_rf))

#trying with naive bayes
CLF_gnb = GaussianNB()
CLF_mnb = MultinomialNB()

CLF_gnb.fit(x_train[:,4:6], y_train)
CLF_mnb.fit(x_train[:,4:6], y_train)

predict_gnb = CLF_gnb.predict(x_test[:,4:6])
predict_mnb = CLF_mnb.predict(x_test[:,4:6])

print("Gaussian Naive Bayes Accuracy: ",accuracy_score(y_test, predict_gnb))
print("Multinomial Naive Bayes Accuracy: ",accuracy_score(y_test, predict_mnb))

#trying with svm
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 42)
CLF_svm = svm.SVC()
CLF_svm.fit(x_train[:,4:6], y_train)
predict_svm = CLF_svm.predict(x_test[:,4:6])
print("SVM Classifier Acc: ", accuracy_score(y_test, predict_svm))
