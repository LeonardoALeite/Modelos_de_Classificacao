
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from pandas import Series, DataFrame
from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix

model1 = KNeighborsClassifier(n_neighbors = 10)
model2 = DecisionTreeClassifier()
model3 = GaussianNB()


url = 'https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv'
titanic = pd.read_csv(url)
titanic.columns = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
titanic_data = titanic.drop(['PassengerId','Name','Ticket','Cabin'], 1)

def age_approx(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

titanic_data['Age'] = titanic_data[['Age', 'Pclass']].apply(age_approx, axis=1)
titanic_data.dropna(inplace=True)
gender = pd.get_dummies(titanic_data['Sex'],drop_first=True)
embark_location = pd.get_dummies(titanic_data['Embarked'],drop_first=True)
titanic_data.drop(['Sex', 'Embarked'],axis=1,inplace=True)
titanic_dmy = pd.concat([titanic_data,gender,embark_location],axis=1)
titanic_dmy.drop(['Pclass'],axis=1,inplace=True)
titanic_dmy.drop(['Q'],axis=1,inplace=True)

X = titanic_dmy.iloc[:,[1,2,3,4,5,6]].values
y = titanic_dmy.iloc[:,0].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)

#Testanto o knn apenas
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(metrics.accuracy_score(y_test, y_pred))


#Testanto Arvore de decisão  apenas
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(metrics.accuracy_score(y_test, y_pred))


#Testanto Arvore de Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(metrics.accuracy_score(y_test, y_pred))

#Esemble baseado em voto marjoritário
model = VotingClassifier(estimators=[('knn', model1), ('dt', model2), ('gnb', model2)], voting='hard')
model.fit(X_train,y_train)
model.score(X_test,y_test)


#Ensemble baseado em pesos e na probabilidade da classes
knn.fit(X_train,y_train)
nb.fit(X_train,y_train)
dt.fit(X_train,y_train)

pred1=knn.predict_proba(X_test)
pred2=nb.predict_proba(X_test)
pred3=dt.predict_proba(X_test)

finalpred=(pred1*0.2+pred2*0.5+pred3*0.3)

for i in range(0, len(finalpred[:,1])):
    if finalpred[i,0] > finalpred[i,1]:
        y_pred[i] = 0
    else:
        y_pred[i] = 1
        
print(metrics.accuracy_score(y_test, y_pred))


#Ramdon Forest
from sklearn.ensemble import RandomForestClassifier
modelRfc= RandomForestClassifier(random_state=1)
modelRfc.fit(X_train, y_train)
modelRfc.score(X_test,y_test)

#Exibe as caracteristicas mais relevantes para o ramdom forest
for i, j in sorted(zip(titanic_dmy.iloc[:,[1,2,3,4,5,6]].columns, modelRfc.feature_importances_)):
    print(i, j)
    
    
#Boosting utilizando  o algoritmo AdaBoost
from sklearn.ensemble import AdaBoostClassifier
modelADB = AdaBoostClassifier(random_state=1, n_estimators = 20)
modelADB.fit(X_train, y_train)
modelADB.score(X_test,y_test)
