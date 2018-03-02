def titanic():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import preprocessing
    from sklearn import linear_model

    dataset_training = pd.read_csv('C:/users/myPC/Desktop/ml/Titanic/train.csv')
    dataset_test = pd.read_csv('C:/users/myPC/Desktop/ml/Titanic/test.csv')
    passenger_id = dataset_test['PassengerId']
    # for the feature `Fare` in the `test_data`, only one missed
    dataset_test.loc[dataset_test.Fare.isnull(), 'Fare'] = 0.0
    # drop the irrelevant features
    dataset_training.drop(labels=['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    dataset_test.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)
    # predict `age` which is missed by others' features
    dataset_training_age = dataset_training[['Pclass', 'SibSp', 'Parch', 'Fare', 'Age']]
    dataset_test_age = dataset_test[['Pclass', 'SibSp', 'Parch', 'Fare', 'Age']]
    age_known0 = dataset_training_age[dataset_training_age.Age.notnull()].as_matrix()  # get the `ndarray`
    age_unknown0 = np.array(dataset_training_age[dataset_training_age.Age.isnull()])
    age_unknown1 = dataset_test_age[dataset_test_age.Age.isnull()].as_matrix()
    training_data_age = age_known0[:, :-1]
    training_target_age = age_known0[:, -1]
    rfr = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=0)  # enable to fit them by the 1000 trees
    rfr.fit(training_data_age, training_target_age)
    predicts = rfr.predict(age_unknown0[:, :-1])
    dataset_training.ix[dataset_training.Age.isnull(), 'Age'] = predicts  # fill the `age` which is missed
    # fit model(RandomForestRegressor) by the `training data`
    dataset_test.loc[dataset_test.Age.isnull(), 'Age'] = rfr.predict(age_unknown1[:, :-1])
    dataset_training.ix[dataset_training.Cabin.notnull(), 'Cabin'] = 'Yes'  # fill the `Cabin` as `Yes` which `notnull`
    dataset_training.ix[dataset_training.Cabin.isnull(), 'Cabin'] = 'No'  # else, `No`
    dataset_test.ix[dataset_test.Cabin.notnull(), 'Cabin'] = 'Yes'
    dataset_test.ix[dataset_test.Cabin.isnull(), 'Cabin'] = 'No'
    # dummy some fields whose types of [`object`, `category`] to eliminate relation between categories
    dataset_training_dummies = pd.get_dummies(dataset_training, columns=['Pclass', 'Sex', 'Cabin', 'Embarked'])
    dataset_test_dummies = pd.get_dummies(dataset_test, columns=['Pclass', 'Sex', 'Cabin', 'Embarked'])
    ss = preprocessing.StandardScaler()  # standardize some features which have some differences
    dataset_training_dummies['Age'] = ss.fit_transform(dataset_training_dummies.Age.reshape(-1, 1))
    dataset_training_dummies['Fare'] = ss.fit_transform(dataset_training_dummies.Fare.reshape(-1, 1))
    dataset_test_dummies['Age'] = ss.fit_transform(dataset_test_dummies.Age.reshape(-1, 1))
    dataset_test_dummies['Fare'] = ss.fit_transform(dataset_test_dummies.Fare.reshape(-1, 1))
    # get all processed samples
    print(dataset_training_dummies)
    dataset_training_dummies = dataset_training_dummies.filter(regex='Age|SibSp|Parch|Fare|Pclass_*|Sex_*|Cabin_*|Embarked_*|Survived').as_matrix()
    # print(data_training_dummies.info())
    training_data = dataset_training_dummies[:, 1:]
    training_target = dataset_training_dummies[:, 0:1]
    lr = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-5)
    lr.fit(training_data, training_target)
    predicts = lr.predict(dataset_test_dummies)
    ans = pd.DataFrame({'PassengerId': passenger_id, 'Survived': predicts.astype(np.int32)})
    # print(ans)
    ans.to_csv('C:/users/myPC/Desktop/ml/Titanic/submission.csv', index=False)  # ignore label-index
    # print(pd.DataFrame({'features': list(dataset_test_dummies[1:]), 'coef': list(lr.coef_.T)}))

if __name__ == '__main__':
	titanic()