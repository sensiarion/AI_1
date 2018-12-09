#Лабораторная работа 4 - Предварительная обработка данных

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sp
import pandas as pd
import random

data = pd.read_csv('./Iris.csv', index_col='Id')
data.head()

### 2. Пропуски в данных

# так как в исходных данных нет пропусков, то создадим их случайным образом
rnd = random.Random('seed_here')

nan_data = data.copy()
for i in range(nan_data.shape[0]):
    if rnd.randint(0, 100) < 10:
        select_column = rnd.choice(nan_data.columns)
        print(select_column, i)
        nan_data[select_column][i] = NaN

print('\nИзначально')
print(data.dropna().count())
print('\nПосле обработки')
print(nan_data.dropna().count())
nan_data.head(55)

## заполнение пропусков 

# заполнение пропусков константой
nan_data_fill_simple = nan_data.fillna(0)
nan_data_fill_simple.head(20)

# удаление пропусков через взятие среднего
nan_data_fill_mean = nan_data.fillna(nan_data.mean())
print(f'Mean: \n{nan_data.mean()}')
nan_data_fill_mean.head(20)

# удаление пропусков через их заполнения соседнеми значениями
# pad - это метод при котором значение берется с соседней ячейки
nan_data_fill_near_value = nan_data.fillna(method='pad')
nan_data_fill_near_value.head(20)

# заполнение пропусков через интерполяцио столбцов с заполнением пропусков
nan_data_interpolate = nan_data.interpolate(method='cubic')
nan_data_interpolate.head(20)
#nan_data_interpolate_fill_near_value = nan_data_interpolate.fillna(method='pad')

# В крайнем случае вы можите просто выкинуть данные с пропусками, но делайте это в самом крайнем случае
nan_data_dropnan = nan_data.dropna()
print(nan_data_dropnan.count())
nan_data_dropnan.head(20)

### 2. Нормализация
# нормализация данных это приведение данных в некий стандартный вид
# для алгоритмов машинного обучения это числовой вид и, желательно, чтобы все числа лежали в диапазоне от -1 до 1 или от 0 до 1. 
# Также тут необходимо убрать выбросы (промахи) в данных
import sklearn.preprocessing as preprocessing

# Первым делом представим строковые данные как числовые
# также отлично подходит для удаления числовой зависимости у классовых значений

number_data = pd.get_dummies(data)
number_data.head()

# Простая номализация.
# Данных маштабируются на промежуток от 0 до 1, где 0 - минимум в данных, а 1 - это максимум.
# Все остальные находятся между ними

mm_scalar = preprocessing.MinMaxScaler()
mm_scalar.fit(number_data)
mm_norm_number_data = pd.DataFrame(columns=number_data.columns, data=mm_scalar.transform(number_data))
mm_norm_number_data.head()
# Данный нормализатор выравнивает данные относительно минимального и максимального
mm_norm_number_data.plot(figsize=(10, 10))

#Следующий нормализатор похож на предыдущий, только он маштабирует данные относительно максимального по модулю
ma_scaler = preprocessing.MaxAbsScaler()
ma_scaler.fit(number_data)
ma_norm_data = pd.DataFrame(columns=number_data.columns, data=ma_scaler.transform(number_data))

# Нормализация на основе дисперсии и мат. ожидания
std_scaler = preprocessing.StandardScaler()
std_scaler.fit(number_data)
std_norm_data = pd.DataFrame(columns=number_data.columns, data=std_scaler.transform(number_data))
# Обратите внимание
# Данные хоть и маштабированы, но они не лежат в диапазоне от -1 до 1, а привышают его. Тут можно испоьзовать дополнительно MinMaxScaler

# Нормализация на основе дисперсии и мат. ожидания c удалением выбросов
std_clear_scaler = preprocessing.RobustScaler()
std_clear_scaler.fit(number_data)
std_clear_norm_data = pd.DataFrame(columns=number_data.columns, data=std_clear_scaler.transform(number_data))

### 4. Pipeline
# Pipeline - это обект который умеет обединять в себе несколько препроцессоров для данных превращая их в один
#Для начала давайте напишем несколько наших препроцессоров

#препроцессор для удаления выбросов написаный руками. Для его реализации будем использовать правило 3-х сигм для нормального распределния
#простой препроцессор для преобразования обекта DataFrame в ndarray

import sklearn.pipeline as pipeline
from sklearn.base import BaseEstimator, TransformerMixin
