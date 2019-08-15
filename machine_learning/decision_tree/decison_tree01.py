#coding=utf-8
from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
import numpy as np

def test01():
    X = [[0, 0], [2, 2], [3,3]]
    y = [0, 1, 0]
    clf = tree.DecisionTreeClassifier()
    clf.fit(X,y)
    print(clf.predict([[1, 1],[-1, -1]]))

    dot_data = tree.export_graphviz(clf, out_file='tree.dot')
    # dot -Tpng tree.dot -o 1.png
    print(clf.score(X, y))


def get_data():
    data = pd.read_csv('../data/watermelon.csv')
    ret_data= pd.DataFrame()
    ret_data['color'] = LabelEncoder().fit_transform(data['色泽'])
    ret_data['root'] = LabelEncoder().fit_transform(data['根蒂'])
    ret_data['knock'] = LabelEncoder().fit_transform(data['敲声'])
    ret_data['texture'] = LabelEncoder().fit_transform(data['纹理'])
    ret_data['umbilical'] = LabelEncoder().fit_transform(data['脐部'])
    ret_data['touch'] = LabelEncoder().fit_transform(data['触感'])
    ret_data['y'] = LabelEncoder().fit_transform(data['好瓜'])

    y = ret_data['y']
    ret_data.drop(['y'], axis=1, inplace = True)
    return ret_data, y

def train(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 123)
    print(len(y_train), len(y_test))
    clf = tree.DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=2)
    clf.fit(X_train, y_train)
    print('train score:', clf.score(X_train, y_train), ' test score:', clf.score(X_test, y_test))
    dot_data = tree.export_graphviz(clf, out_file='1.dot',feature_names=X_train.columns.values, class_names=['good', 'bad'])


def SearchParms(x_train, y_train):
    dtc = tree.DecisionTreeClassifier()
    parameters = {'min_samples_split': range(2, 10, 1), 'min_samples_leaf': [1, 2, 3, 4]}
    clf = GridSearchCV(estimator=dtc, param_grid=parameters, cv=5)
    clf.fit(x_train, y_train)
    print("DecisionTree", clf.best_params_, clf.best_score_)

if __name__=="__main__":
    x, y = get_data()
    train(x, y)
    SearchParms(x, y)