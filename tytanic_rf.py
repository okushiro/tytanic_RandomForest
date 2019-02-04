import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#ディレクトリの設定
os.chdir('/Users/okushirokentaro/Desktop/kaggle/tytanic/')

#データの読み込み
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#欠損値処理
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna('S')

test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Embarked'] = test['Embarked'].fillna('S')

#カテゴリ変数の変換
train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 0)
train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

test['Sex'] = test['Sex'].apply(lambda x: 1 if x == 'male' else 0)
test['Embarked'] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#学習データとテストデータの作成
train = train.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)
X_train = train.drop('Survived', axis=1)
Y_train = train.Survived
testdata = test.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)

#ランダムフォレスト
forest = RandomForestClassifier(random_state=1234,criterion="gini",max_depth=10)
fit = forest.fit(X_train, Y_train)
Survived = fit.predict(testdata)

#提出データ
submit = pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':Survived})
submit.to_csv("submit.csv",index=False)
