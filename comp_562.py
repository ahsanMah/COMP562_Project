import pandas as pd
import numpy as np
from sklearn.metrics import classification_report as report
from sklearn.linear_model import LogisticRegression
RANDOM_STATE = 42
np.random.seed(seed=RANDOM_STATE)


http_train=pd.read_csv('data/http_train.csv',sep=',')
http_test=pd.read_csv('data/http_test.csv',sep=',')

#train_raw_data = http_train.drop(http_train.index[0])
train_data = http_train.drop(http_train.columns[-1],axis='columns')
train_label = http_train.iloc[:,-1]

#test_raw_data = http_test.drop(http_test.index[0])
test_data = http_test.drop(http_test.columns[-1],axis='columns')
test_label = http_test.iloc[:,-1]

#train_data.head()
http_train.head()


lg_model=LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr')
lg_model.fit(train_data,train_label)

result_model=lg_model.predict(test_data)
report_model=report(test_label,result_model,digits=5)

print(report_model)