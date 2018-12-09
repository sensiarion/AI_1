#Лабораторная работа 4 - Предварительная обработка данных

import warnings
warnings.simplefilter('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sp
import pandas as pd
import random
import sklearn.preprocessing as preprocessing

data = pd.read_csv('../../../../data/titanic_train.csv', index_col='PassengerId').drop(['Ticket', 'Cabin', 'Name'], axis=1)

#print(data.head())
#print(data.shape)
#print(data.isna().sum())

#data['Age'].plot() # функция для интерполяции
# распределение пропусков для pad
#plt.axis('off')
#plt.hlines(1,1, data.index.values[-1])
#plt.eventplot(data[data['Age'].isna()].index.values, orientation='horizontal', colors='b')
# подходит хорошо
data = data.fillna(method='pad')

#########################################################################################



### Пропуски в данных

rnd = random.Random('seed_here')
nan_data = data.copy()
for i in range(nan_data.shape[0]):
    if rnd.randint(0, 100) < 20:
        select_column = rnd.choice(nan_data.columns)
        #print(select_column, i)
        nan_data[select_column][i] = np.NaN

print('\nДанных без пропусков изначально:')
print(data.dropna().count())
print('\nПосле обработки:')
print(nan_data.dropna().count())

print(nan_data.head(10))

print("\nзаполнение пропусков константой")
nan_data_fill_simple = nan_data.fillna(0)
print(nan_data_fill_simple.head(10))

print("\nудаление пропусков через взятие среднего")
nan_data_fill_mean = nan_data.fillna(nan_data.mean())
print(f'Mean: \n{nan_data.mean()}')
print(nan_data_fill_mean.head(10))

print("\nудаление пропусков через их заполнение соседними значениями")
nan_data_fill_near_value = nan_data.fillna(method='pad')
print(nan_data_fill_near_value.head(10))

print("\nпросто удаление данных с пропусками (крайний случай)")
nan_data_dropna = nan_data.dropna()
print(nan_data_dropna.head(10))



#########################################################################################

# удаление числовой зависимости у классовых занчений и обработка строковых данных
data_y = data['Pclass']
data_X = pd.get_dummies(data.drop('Pclass', axis=1), columns=['Sex', 'Embarked'])
print(data_X.head())

#########################################################################################



### Нормализация
number_data = data_X.copy()

# масштаб. от 0 до 1
mm_scalar = preprocessing.MinMaxScaler()
mm_scalar.fit(number_data)
mm_norm_number_data = pd.DataFrame(columns=number_data.columns, data=mm_scalar.transform(number_data))
mm_norm_number_data.plot()

ma_scaler = preprocessing.MaxAbsScaler()
ma_scaler.fit(number_data)
ma_norm_data = pd.DataFrame(columns=number_data.columns, data=ma_scaler.transform(number_data))
ma_norm_data.plot()

# с учетом дисперсии и мат ожидания
std_scaler = preprocessing.StandardScaler()
std_scaler.fit(number_data)
std_norm_data = pd.DataFrame(columns=number_data.columns, data=std_scaler.transform(number_data))
std_norm_data.plot()

# + без выбросов
std_clear_scaler = preprocessing.RobustScaler()
std_clear_scaler.fit(number_data)
std_clear_norm_data = pd.DataFrame(columns=number_data.columns, data=std_clear_scaler.transform(number_data))
std_clear_norm_data.plot()

### Pipeline

import sklearn.pipeline as pipeline
from sklearn.base import BaseEstimator, TransformerMixin

#(РодительскийКласс)
class DropOutlet(BaseEstimator):
    """
    Удаление выбросов, основываясь на правиле 3-х сигм (только для нормального распределения)
    """

    # переопределение виртуальных методов родителя
    #var: Type
    def fit(self, X: pd.DataFrame, y=None):
        #data.select_dtypes(include=['float', 'int']) #WHAT
        #self.std = X.std()
        self.std = X.select_dtypes(include=['float', 'int']).std()
        self.columns = self.std.index.values
        return self

    #-> ReturnType
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Убираем все здачения, котоые находятся дальше 3-х стандартных отконений (сигма) от мат. ожидания случайной величины
        """
        X_cols = X[self.columns]
        return X[X.index.isin(X_cols[ (X_cols - X_cols.mean()).abs() <= 3*self.std ].dropna().index)]

class PandasToNumpy(BaseEstimator):
    """
    Просто преобразует данные из DataFrame от pandas к матрице от numpy (ndarray)
    """

    def fit(self, X : pd.DataFrame = None, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return X.values

class SparseToArray(TransformerMixin, BaseEstimator):
    """
    Класс преобразует sparse matrix в ndarray
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.toarray()


class ModifyLabelEncoder(preprocessing.LabelEncoder):
    """
    Стандартный костыль для препроцессора LabelEncoder
    """

    def fit(self, X, y=None):
        return super().fit(X)

    def transform(self, X, y=None):
        print(1)
        return super().transform(X)

    def fit_transform(self, X, y=None):
        return super().fit_transform(X)

X_data = data_X.copy()
Y_data = data_y.copy()

pipe_outlet = pipeline.Pipeline([
    ('drop_outlet', DropOutlet()),
])
outletless_data_X = pipe_outlet.fit_transform(X_data)
outletless_data_Y = Y_data[ Y_data.index.isin(outletless_data_X.index) ]

pipe_x = pipeline.Pipeline([
    ('to_matrix', PandasToNumpy()),
    ('norm', preprocessing.MinMaxScaler())
])
pipe_y = pipeline.Pipeline([
    ('to_matrix', PandasToNumpy()),
    ('label_encoder', ModifyLabelEncoder()),
])
number_data_ready_X = pipe_x.fit_transform(outletless_data_X)
number_data_ready_Y = pipe_y.fit_transform(outletless_data_Y)

print(number_data_ready_X.shape, number_data_ready_Y.shape)
print(number_data_ready_X[:5,:])
print(number_data_ready_Y)

#########################################################################################
#Standardization of a dataset is a common requirement for many machine learning estimators. Typically this is done by removing the mean and scaling to unit variance. However, outliers can often influence the sample mean / variance in a negative way. In such cases, the median and the interquartile range often give better results.

scaler = preprocessing.RobustScaler()
scaler.fit(number_data)
data_X = pd.DataFrame(columns=data_X.columns, data=scaler.transform(data_X))
print(data_X.values[:5,:])
print(data_y.values)
#########################################################################################

plt.show()
