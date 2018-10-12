import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def preProcessData(file_name, drop_array, data_type):
    data = pd.read_csv(file_name)

    # Convert month and year to int
    data['month'] = pd.to_datetime(data['month'])
    data['year'] = pd.DatetimeIndex(data['month']).year
    data['month'] = pd.DatetimeIndex(data['month']).month

    if ('private' in file_name):
        # Convert tenure to int
        data.loc[data.tenure != 'Freehold', 'tenure'] = 0
        data.loc[data.tenure == 'Freehold', 'tenure'] = 1
        data['tenure'] = pd.to_numeric(data['tenure'])
        
        # Fill NaN floor numbers with the mean floor number
        data['floor_num'] = data['floor_num'].fillna(round(data['floor_num'].mean()))
        
        # Convert completion_date to int
        data.loc[data.completion_date == 'Uncompleted', 'completion_date'] = "1990"
        data.loc[data.completion_date == 'Uncomplete', 'completion_date'] = "1990"
        data.loc[data.completion_date == 'Unknown', 'completion_date'] = "1990"
        data['completion_date'] = data.completion_date.str[-4:]
        data['completion_date'] = pd.to_numeric(data['completion_date'])
    
    if ('hdb' in file_name):
        # Convert flat_type to int
        data['flat_type'] = data['flat_type'].astype('category')
        data['flat_type'] = data['flat_type'].cat.reorder_categories(['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'MULTI GENERATION', 'EXECUTIVE'], ordered=True)
        data['flat_type'] = data['flat_type'].cat.codes
    
    data = data.drop(drop_array, axis=1) # Drop useless features
    data['data_type'] = data_type # Add an artificial column to differentiate train from test set
    return data
    

def formatData(train_filename, test_filename, drop_array, price):
    train = preProcessData(train_filename, drop_array, 'train')
    test = preProcessData(test_filename, drop_array, 'test')
    data = pd.concat([train, test])
    data = pd.get_dummies(data) # Convert categorical variables into dummy variables
    
    train = data[data['data_type_train'] == 1]
    test = data[data['data_type_test'] == 1]
    train = train.drop(['data_type_train', 'data_type_test'], axis=1)
    test = test.drop(['data_type_train', 'data_type_test', price], axis=1)
    
    if ('private' in train_filename):
        train = train.drop(train[train.price < 10000].index) # Remove data with obviously wrong prices
    
    x_train = train.drop(price, axis=1)
    y_train = train[price]
    return x_train, y_train, test


# Build the model
def randomForestRegressor(x_train, y_train, test, n_estimators, min_samples_leaf):
    clf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf)
    clf.fit(x_train, y_train)
    return clf.predict(test)


# Create a csv file containing the prediction results
def createCsv(prediction):
    with open('prediction.csv', 'w') as predictionFile:
        wr = csv.writer(predictionFile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        wr.writerow(['index', 'price'])
        for i in range(len(prediction)):
            wr.writerow([i, round(prediction[i])])


hdb_x_train, hdb_y_train, hdb_test = formatData('hdb_train.csv', 'hdb_test.csv', ['flat_model', 'town', 'index', 'block', 'storey_range', 'street_name'], 'resale_price')
hdb_prediction = randomForestRegressor(hdb_x_train, hdb_y_train, hdb_test, 200, 5)


private_x_train, private_y_train, private_test = formatData('private_train.csv', 'private_test.csv', ['area', 'index', 'project_name', 'address', 'contract_date', 'postal_district', 'postal_sector', 'unit_num'], 'price')
private_prediction = randomForestRegressor(private_x_train, private_y_train, private_test, 200, 1)

createCsv(np.append(hdb_prediction, private_prediction))
