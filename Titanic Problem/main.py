import pandas as pd
from sklearn.model_selection import train_test_split


dataset = ("train.csv")
my_data = pd.read_csv(dataset)

my_data = my_data.drop(['Ticket', 'Cabin', 'Name'], axis=1)
my_data = my_data.dropna()
my_data.head(5)
def transform_data(df):
    Sex_dummies = pd.get_dummies(df['Sex'], prefix='Sex')
    Embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    Pclass_dummies = pd.get_dummies(df['Pclass'], prefix='Pclass')
    
    df = df.drop(['Sex', 'Embarked', 'Pclass'], axis=1)
    df = pd.concat([df, Sex_dummies, Embarked_dummies, Pclass_dummies], axis=1)
    
    return df

my_data = transform_data(my_data)
labels = my_data['Survived']
my_data = my_data.drop(['Survived'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(my_data, labels, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(X_train, y_train)