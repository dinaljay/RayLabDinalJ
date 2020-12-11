#Recursive feature elimination

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV

logreg = LogisticRegression(random_state=0)

rfe = RFECV(logreg,min_features_to_select=35,cv=4)
rfe = rfe.fit(X_scaled, y)

cols =[]
i=0
for value in rfe.ranking_:
    if value==1:
        cols.append(i)
    i+=1

cols_to_drop=[]
columns = X.columns

for i in cols:
    cols_to_drop.append(columns[i])

X = X.drop(cols_to_drop, axis=1)

#Rescale data
#X_scaled = preprocessing.scale(X)