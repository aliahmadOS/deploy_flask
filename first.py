import sklearn
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import pickle
import os


data=sklearn.datasets.load_iris()

train_data,test_data,train_label,test_label=sklearn.model_selection.train_test_split(data.data,data.target, random_state=123, test_size=0.3)
# print(train_data,train_label)

model=sklearn.ensemble.RandomForestClassifier(n_estimators=500)

model.fit(train_data,train_label)

result=model.score(test_data,test_label)

# print(result)

filename='iris_model.pkl'

pickle.dump(model,open(filename,'wb'))
