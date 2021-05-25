import numpy as np
import pandas as pd
import pydot as pydot
from sklearn.tree import DecisionTreeClassifier, export_graphviz

def get_data(path1, path2):

    # 파일 불러오기
    train = pd.read_csv(path1)
    test = pd.read_csv(path2)

    # 데이터 전처리

    # male은 0으로, female은 1로 변경
    train.loc[train["Sex"] == "male", "Sex"] = 0
    train.loc[train["Sex"] == "female", "Sex"] = 1

    test.loc[test["Sex"] == "male", "Sex"] = 0
    test.loc[test["Sex"] == "female", "Sex"] = 1

    # Embarked 항목의 데이터에 대해 One Hot Incoding
    train["Embarked_C"] = train["Embarked"] == "C"
    train["Embarked_S"] = train["Embarked"] == "S"
    train["Embarked_Q"] = train["Embarked"] == "Q"

    test["Embarked_C"] = test["Embarked"] == "C"
    test["Embarked_S"] = test["Embarked"] == "S"
    test["Embarked_Q"] = test["Embarked"] == "Q"
    print(train)

    # test 데이터의 Fare 항목에 있는 빈칸에 0 채우기
    test.loc[pd.isnull(test["Fare"]), "Fare"] = 0

    # 예측할 때 고려할 요소들
    features = ["Sex", "Pclass", "SibSp", "Parch", "Fare",
                "Embarked_C", "Embarked_Q", "Embarked_S"]

    X_train = train[features]

    x_test = test[features]

    # 정답 항목인 Survived 데이터를 y_train에 저장
    y_train = train["Survived"]

    run_dt(X_train, y_train, x_test, features)

def run_dt(X_train, y_train, x_test, features) :

    # Decision Tree 만들기
    seed = 0
    dt = DecisionTreeClassifier(max_depth = 30, random_state = seed)

    # Decision Tree 학습시키기
    dt.fit(X_train, y_train)

    # 시각화, png 파일로 저장
    export_graphviz(dt, out_file="dt.dot", class_names=['No', 'Yes'], 
                   feature_names=features, impurity=False, filled=True)
    (graph, ) = pydot.graph_from_dot_file('dt.dot', encoding='utf8')
    graph.write_png('dt.png')

    # 결과 예측
    result = dt.predict(x_test)

    # 예측된 결과를 submission 형식에 맞게 작성
    submission = pd.read_csv('gender_submission.csv', index_col='PassengerId')
    submission["Survived"] = result

    # 결과를 csv 파일로 저장
    submission.to_csv('201802161_submission.csv')

def main():
    get_data('train.csv', 'test.csv')

if __name__ == '__main__':
    main()