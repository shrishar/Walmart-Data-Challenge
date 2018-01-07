import pandas as pd
from pandas import Series,DataFrame
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
import pylab
from sklearn import linear_model
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.metrics import r2_score

# machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def hotencode(data):
    data['p_1'] = data.PromoInterval.apply(lambda x: x[:3] if type(x) == str else 0)
    data['p_2'] = data.PromoInterval.apply(lambda x: x[4:7] if type(x) == str else 0)
    data['p_3'] = data.PromoInterval.apply(lambda x: x[8:11] if type(x) == str else 0)
    data['p_4'] = data.PromoInterval.apply(lambda x: x[12:15] if type(x) == str else 0)
    data = pd.get_dummies(data, columns=['p_1', 'p_2', 'p_3', 'p_4','StateHoliday','StoreType','Assortment'])
    data.drop(['PromoInterval','p_1_0', 'p_2_0', 'p_3_0', 'p_4_0','StateHoliday_0','year'], axis=1, inplace=True)
    data = data.fillna(0)
    data = data.sort_index(axis=1)
    return data

def datebreakdown(data):
    data['year'] = data.Date.apply(lambda x: x.year)
    data['month'] = data.Date.apply(lambda x: x.month)
    data['woy'] = data.Date.apply(lambda x: x.weekofyear)
    data.drop(['Date'], axis=1, inplace=True)
    return data


def preprocessing(data,store):
    data = data[data['Open'] != 0]
    data = data.merge(store, on='Store', copy=False)
    data= datebreakdown(data)
    data['CompetitionOpen'] = 12 * (data.year - data.CompetitionOpenSinceYear) + \
                              (data.month - data.CompetitionOpenSinceMonth)
    data['CompetitionOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)
    data.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], axis=1,
              inplace=True)

    data['PromoOpen'] = 12 * (data.year - data.Promo2SinceYear) + \
                        (data.woy - data.Promo2SinceWeek) / float(4)
    data['PromoOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)
    data.drop(['Promo2SinceYear', 'Promo2SinceWeek'], axis=1,
              inplace=True)
    data=hotencode(data)
    return data


def correlation(store_piv,start_store,end_store):
    fig, (axis1) = plt.subplots(1, 1, figsize=(15, 5))
    sns.heatmap(store_piv[list(range(start_store, end_store + 1))].corr(), annot=True, linewidths=2,ax=axis1)
    plt.show()

# Visualization functions:
# Sales and customers mean over the three years:

def SalesCustPlot(data):
    data['Date'] = data['Date'].apply(lambda x: (str(x)[:7]))
    sales_avg = data.groupby('Date')["Sales"].mean()
    percent_change = data.groupby('Date')["Sales"].sum().pct_change()
    fig, (axis1, axis2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8))
    ax1 = sales_avg.plot(legend=True, ax=axis1, marker='o', title="Average Sales")
    ax1.set_xticks(range(len(sales_avg)))
    ax1.set_xticklabels(sales_avg.index.tolist(), rotation=90)
    ax2 = percent_change.plot(legend=True, ax=axis2, marker='o', rot=90, colormap="summer",
                              title="Sales Percent Change")
    plt.show()

def mergeStoreSalesCust(data,store):
	avg_sale_cust = data.groupby('Store')[["Sales", "Customers"]].mean()
	sale_cust_df = DataFrame({'Store':avg_sale_cust.index,
                      'Sales':avg_sale_cust["Sales"], 'Customers': avg_sale_cust["Customers"]},
                      columns=['Store', 'Sales', 'Customers'])
	combined_store = pd.merge(sale_cust_df, store, on='Store')
	return combined_store


def SalesCustperYearPlot(data):
    fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
    sns.barplot(x='Year', y='Sales', data=data, ax=axis1)
    sns.barplot(x='Year', y='Customers', data=data, ax=axis2)
    plt.show()

def CustomersSalesPerDay(data):
    fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))

    sns.barplot(x='DayOfWeek', y='Sales', data=data, order=[1, 2, 3, 4, 5, 6, 7], ax=axis1)
    sns.barplot(x='DayOfWeek', y='Customers', data=data, order=[1, 2, 3, 4, 5, 6, 7], ax=axis2)
    plt.show()

def withwithoutpromo(data):
    fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
    sns.barplot(x='Promo', y='Sales', data=data, ax=axis1)
    sns.barplot(x='Promo', y='Customers', data=data, ax=axis2)
    plt.show()

def stateholiday(data):
    data["StateHoliday"] = data["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})
    fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
    sns.barplot(x='StateHoliday', y='Sales', data=data, ax=axis1)
    sns.barplot(x='StateHoliday', y='Customers', data=data, ax=axis2)
    plt.show()

def competitiondistanceplot(data):
    data["CompetitionDistance"].fillna(data["CompetitionDistance"].median())
    data.plot(kind='scatter', x='CompetitionDistance', y='Sales', figsize=(15, 4))
    plt.show()


def schoolholiday(data):
    sns.countplot(x='SchoolHoliday', data=data)
    fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
    sns.barplot(x='SchoolHoliday', y='Sales', data=data, ax=axis1)
    sns.barplot(x='SchoolHoliday', y='Customers', data=data, ax=axis2)
    plt.show()

def OverallSalesCustforStoreType(data):
    sns.countplot(x='StoreType', data=data, order=['a', 'b', 'c', 'd'])
    fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
    sns.barplot(x='StoreType', y='Sales', data=data, order=['a', 'b', 'c', 'd'], ax=axis1)
    sns.barplot(x='StoreType', y='Customers', data=data, order=['a', 'b', 'c', 'd'], ax=axis2)
    plt.show()

def CompetetionsEffect(data,combined_store,store_id):
    store_data = data[data["Store"] == store_id]
    store_data['Date'] = store_data['Date'].apply(lambda x: (str(x)[:7]))
    average_store_sales = store_data.groupby('Date')["Sales"].mean()
    y = combined_store["CompetitionOpenSinceYear"].loc[combined_store["Store"] == store_id].values[0]
    m = combined_store["CompetitionOpenSinceMonth"].loc[combined_store["Store"] == store_id].values[0]
    # Plot
    fig, (axis1) = plt.subplots(1, 1, sharex=True, figsize=(15, 5))
    ax = average_store_sales.plot(legend=True, figsize=(15, 4), marker='o',ax=axis1)
    ax.set_xticks(range(len(average_store_sales)))
    ax.set_xticklabels(average_store_sales.index.tolist(), rotation=90)
    plt.show()

def getTrainTest(train,test,store):
    train = preprocessing(train, store)
    X_train = train.drop(['Sales', 'Customers'], axis=1)
    y_train = train.Sales
    test = preprocessing(test, store)
    for col in train.columns:
        if col not in test.columns:
            test[col] = np.zeros(test.shape[0])

    print('Test data loaded and processed')
    X_test = test.drop(['Sales', 'Customers'], axis=1).values

    return X_train,y_train,X_test


def RandomForest(X_train,y_train,X_test):
    # Fit random forest model
    rf = RandomForestRegressor(n_jobs=-1,n_estimators=40)
    rf.fit(X_train, y_train)
    print('Training data model fit done')

    y_test = rf.predict(X_test)
    return y_test

def RidgeRegressor(X_train,y_train,X_test):
    clf = Ridge(alpha=1.0)
    clf.fit(X_train,y_train)
    y_test=clf.predict(X_test)
    return y_test

def LassoRegressor(X_train,y_train,X_test):
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X_train,y_train)
    y_test= clf.predict(X_test)
    return y_test

def score(y_pred,y_true):
    score=-1*r2_score(y_true, y_pred)
    return score
def OverallSalesCustforAssortment(data):
    sns.countplot(x='Assortment', data=data, order=['a', 'b', 'c'])
    fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
    sns.barplot(x='Assortment', y='Sales', data=data, order=['a', 'b', 'c'], ax=axis1)
    sns.barplot(x='Assortment', y='Customers', data=data, order=['a', 'b', 'c'], ax=axis2)
    plt.show()

def customer(data):
    fig, (axis1, axis2) = plt.subplots(2, 1, figsize=(15, 8))
    sns.boxplot([data["Customers"]], whis=np.inf, ax=axis1)
    data['Date'] = data['Date'].apply(lambda x: (str(x)[:7]))
    average_customers = data.groupby('Date')["Customers"].mean()
    ax = average_customers.plot(legend=True, marker='o', ax=axis2)
    ax.set_xticks(range(len(average_customers)))
    xlabels = ax.set_xticklabels(average_customers.index.tolist(), rotation=90)
    plt.show()

def main():
    walmartdata = pd.read_csv("C:\\college\\Sem 3\\Data challenge\\Data challenge\\sales_cust.csv",parse_dates = ['Date'])
    store = pd.read_csv("C:\\college\\Sem 3\\Data challenge\\Data challenge\\store.csv")
    '''
    Training data created - sales_cust is divided into Jan 2013 from May 2015
    Testing data created  - test data created from June 2015 to July 2015
    '''
    train = walmartdata[(walmartdata['Date'] > '2013-01-01') & (walmartdata['Date'] <= '2015-05-31')]
    test_actual = walmartdata[(walmartdata['Date'] > '2015-06-01') & (walmartdata['Date'] <= '2015-07-31')]
    test = walmartdata[(walmartdata['Date'] > '2015-06-01') & (walmartdata['Date'] <= '2015-07-31')]


    # combined_store=mergeStoreSalesCust(train,store)
    # SalesCustperYearPlot(train)
    # SalesCustPlot(train)
    # CustomersSalesPerDay(train)
    # withwithoutpromo(train)
    # schoolholiday(train)
    # stateholiday(train)
    # OverallSalesCustforStoreType(combined_store)
    # OverallSalesCustforAssortment(combined_store)
    # customer(train)
    # competitiondistanceplot(combined_store)
    # CompetetionsEffect(train,combined_store,6)
    # CompetetionsEffect(train, combined_store, 256)
    # CompetetionsEffect(train, combined_store, 2)
    # CompetetionsEffect(train, combined_store, 1)
    # CompetetionsEffect(train, combined_store, 900)
    y_true=test_actual[test_actual["Open"] != 0]
    y_true=y_true["Sales"]
    X_train,y_train,x_test=getTrainTest(train,test,store)
    y_pred=RandomForest(X_train,y_train,x_test)

    score1=score(y_pred,y_true)
    y_pred1 = RidgeRegressor(X_train, y_train, x_test)
    score2 = score(y_pred1, y_true)
    y_pred3 = LassoRegressor(X_train, y_train, x_test)
    score3 = score(y_pred3, y_true)


main()
