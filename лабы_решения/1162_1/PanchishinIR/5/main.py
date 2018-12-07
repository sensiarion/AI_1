import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sp

import pandas as pd

import sklearn.pipeline as pipeline
import sklearn.preprocessing as preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
#from keras.utils import to_categorical

### load

data = pd.read_csv('../../../../data/titanic_train.csv', index_col='PassengerId').drop(['Ticket', 'Cabin', 'Name'], axis='columns')

print(f'Всего данных: {data.shape[0]}')
print(data.head())

### pipeline pieces (preprocessors)

class FillNa(TransformerMixin, BaseEstimator):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return X.fillna(method='pad')

class DropOutlet(TransformerMixin, BaseEstimator): 
    def fit(self, X: pd.DataFrame, y=None):
        self.std = X.std()
        self.columns = self.std.index.values
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_cols = X[self.columns]
        return X[(X_cols - X_cols.mean()).abs() <= 3*self.std].dropna()

class GetArray(BaseEstimator):
    def fit(self, X : pd.DataFrame = None, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return X.values

class ModifyLabelEncoder(preprocessing.LabelEncoder):
    def fit(self, X, y=None):
        return super().fit(X)
    
    def transform(self, X, y=None):
        return super().transform(X)
    
    def fit_transform(self, X, y=None):
        return super().fit_transform(X)

class ExpandDims(TransformerMixin, BaseEstimator):
    """
    inserting a new axis at the specified position
    """

    def __init__(self, axis):
        self.axis = axis

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.expand_dims(X, axis=self.axis)

###

data_y = data['Pclass']
data_X = pd.get_dummies(data.drop('Pclass', axis=1), columns=['Sex', 'Embarked'])
# X is number data now
print(data_X.columns)


fill_pl = pipeline.Pipeline([
    ('na', FillNa()),
])

outlet_pl = pipeline.Pipeline([
    ('fill', fill_pl),
    ('drop outlet', DropOutlet()),
])

data_X = outlet_pl.fit_transform(data_X)
data_y = data_y[fill_pl.fit_transform(data_y).index.isin(data_X.index)]


normX_pl = pipeline.Pipeline([
    ('array', GetArray()),
    ('mormalize', preprocessing.MinMaxScaler())
])

normy_pl = pipeline.Pipeline([
    ('array', GetArray()),
    ('encode labels', ModifyLabelEncoder()),
    ('add dim', ExpandDims(axis=1)),
    ('one hot or dummy', preprocessing.OneHotEncoder(sparse=False))
])

data_X = normX_pl.fit_transform(data_X)
data_y = normy_pl.fit_transform(data_y)

print(data_X.shape, data_y.shape)
print(data_X)
print(data_y)

### нейросеть

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras import activations
from keras.optimizers import Adam, RMSprop
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy

model = Sequential()
model.add(Dense(10, activation=activations.relu, input_dim=data_X.shape[1]))
model.add(Dense(data_y.shape[1], activation=activations.softmax))
model.compile(Adam(), categorical_crossentropy, metrics=[categorical_accuracy])

#model.fit(data_X, data_y, verbose=2, epochs=100, batch_size=3)
#print(model.evaluate(data_X, data_y))

### перемешаем и получим тестовую выборку

data_Xy = np.hstack((data_X, data_y))
np.random.shuffle(data_Xy)

point_split = round(data_Xy.shape[0] * 0.8)
train_data = data_Xy[:point_split]
test_data = data_Xy[point_split:]

print(f'Всего данных: {data_Xy.shape[0]}')
print(f'Тренировочных данных: {train_data.shape[0]}')
print(f'Тестовых данных: {test_data.shape[0]}')

# Делим все на входные и выходные данные
# Тренировочные
X_train = train_data[:, :data_X.shape[1]]
Y_train = train_data[:, data_X.shape[1]:]
print(f'Размер данных для тренировки (входных): {X_train.shape}')
print(f'Размер данных для тренировки (выходных): {Y_train.shape}')
print(X_train)
print(Y_train)

# Тестовые
X_test = test_data[:, :data_X.shape[1]]
Y_test = test_data[:, data_X.shape[1]:]
print(f'Размер данных для теста (входных): {X_test.shape}')
print(f'Размер данных для теста (выходных): {Y_test.shape}')

###

def make_nn(input_len, output_len):
    model = Sequential()
    model.add(Dense(10, activation=activations.relu, input_dim=input_len))
    model.add(Dense(output_len, activation=activations.softmax))
    model.compile(Adam(), categorical_crossentropy, metrics=[categorical_accuracy])
    return model

model = make_nn(X_train.shape[1], Y_train.shape[1])

from keras import callbacks

#print('Начинаем обучение сети')
#history = model.fit(
#    x=X_train,
#    y=Y_train,
#    batch_size=3,
#    epochs=100,
#    verbose=1,
#    validation_data=(X_test, Y_test),
#    callbacks=[
#        callbacks.History(),
#    ]
#)

#print('Начинаем обучение сети')
#history = model.fit(
#    x=data_X,
#    y=data_y,
#    batch_size=3,
#    epochs=100,
#    verbose=1,
#    #Вот это разделения данных в соотношении 80/20
#    validation_split=0.2,
#    callbacks=[
#        callbacks.History(),
#    ]
#)

#print('Начинаем обучение сети')
#history = model.fit(
#    x=number_data_ready_X,
#    y=number_data_ready_Y,
#    batch_size=3,
##     Так как у нас теперь есть рання остановка мы можем увеличеть количество эпох (было 100 стало 300)
#    epochs=300,
#    verbose=1,
##     Вот это разделения данных в соотношении 80/20
#    validation_split=0.2,
#    callbacks=[
#        callbacks.History(),
#        callbacks.EarlyStopping(
##             На основе какого значения будет приниматься решеня об остановке
#            monitor='val_categorical_accuracy',
##             Указываем направления лучшего значения (min, max, auto) лучшим является если тестовая точноть будет максимальной
#            mode='max',
##             Количество эпок в резельтате которых если значение не изменилось, то произвести остановку
#            patience=50,
##             "Чуствительность" метода - изменения ниже данного значения дубут считатья 0 (изменения в отслеживаемом значении нету)
#            min_delta=0.01,
#        )
#    ]

#В обучении НС самый последний результат обучения не всегда самый лучший результат, это вызвано несколькими причинами, например переобучением сети или скатыванию к среднему значению.
#Для того что бы поймать тот самый лучший результат обучения, были придуманы "контрольные точки" ModelCheckpoint

from pathlib import Path
from os import remove

file_name = 'nn_model_loss-{loss:.2f}_val_loss-{val_loss:.2f}_acc-{categorical_accuracy:.2f}_val_acc-{val_categorical_accuracy:.2f}.hdf5'
def make_save_points(name='save_points', file_name=file_name):
#     выбираем катагол (подробнее https://docs.python.org/3.6/library/pathlib.html ) 
    checkpoints_dir = Path('./').joinpath('save_points')
    print(f'Текущий каталог с контрольными точками {checkpoints_dir.absolute()}')
    # Создаем каталог если его нету
    checkpoints_dir.mkdir(exist_ok=True)
    # Удаляем все из каталога
    for item in checkpoints_dir.iterdir():
        if item.is_file():
            print(f'Удаляем файл {item}')
            remove(item)
    return str(checkpoints_dir.joinpath(file_name))


print('Начинаем обучение сети')
history = make_nn(X_train.shape[1], Y_train.shape[1]).fit(
    x=X_train,
    y=Y_train,
    batch_size=3,
    epochs=300,
    validation_split=0.2,
    verbose=1,
    callbacks=[
        callbacks.History(),
        # Добавляем сценарий ранней остановки
        # Если в течении 30 эпох точность не вырастить более чем на 1%, то произойдет остановка
        callbacks.EarlyStopping(
            monitor='val_categorical_accuracy',
            mode='max',
            patience=50,
            min_delta=0.01
        ),
#         Callback сохранений состояний сети
        callbacks.ModelCheckpoint(
#             Указываем путь для сохранения и формат имен файлов
            make_save_points(file_name=file_name),
#             Указываем какое значение отслеживать
            monitor='val_categorical_accuracy',
#             Указываем, что сохранять надо только лучшие результаты
            save_best_only=True,
#             Говорим как часто проверять, что текущий результат лучше предидущего (в эпохах)
            period=5,
#             Указываем сторону лучших значений
            mode='max'
        )
    ]
)

plt.figure(0, figsize=(10,5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('График ошибки')
plt.ylabel('Значение')
plt.xlabel('Эпоха')
plt.legend(['Ошибка (train)', 'Ошибка (test)']);

plt.figure(1, figsize=(10,5))
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.ylabel('Значение')
plt.xlabel('Эпоха')
plt.title('График точности')
plt.legend(['Точность (train)', 'Точность (test)']);

from keras.models import load_model

import re

# Загружаем контрольную точку (модель)
mx_acc = 0.00;
checkpoints_dir = Path('./').joinpath('save_points')
for item in checkpoints_dir.iterdir():
    if item.is_file():
        res = re.search(r'val_acc-(\d\.\d)', str(item))
        if res is not None:
            cur_acc = float(res.group(1))
            if cur_acc > mx_acc:
                mx_acc = cur_acc
                fname = str(item)

loaded_model = load_model(fname)

# Прдсказываем класс
print(f'Точность предсказания на тренировочных данных {loaded_model.evaluate(X_train, Y_train)[1]}')
print(f'Точность предсказания на тестовых данных {loaded_model.evaluate(X_test, Y_test)[1]}')
predict = loaded_model.predict_classes(X_test)
print(predict)

# Преобразуем классы в виде числа в их оригеналы
print(normy_pl.named_steps['encode labels'].inverse_transform(predict))

# Прдсказываем вероятности для классов
print(loaded_model.predict(X_train))

###

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf(xx, yy)
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

from sklearn.decomposition import PCA
pipe_y_mini = pipeline.Pipeline([
    ('to_matrix', GetArray()),
    ('label_encoder', ModifyLabelEncoder()),
])

#Y = pipe_y_mini.fit_transform(data_y)
Y = data_y
X = data_X

pca = PCA(n_components=2)

_X = pca.fit_transform(X)
print(_X.shape)

clf_predict = lambda xx, yy: model.predict_classes(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))

X0, X1 = _X[:, 0], _X[:, 1]
xx, yy = make_meshgrid(X0, X1)

clf_predict(xx, yy)

fig, ax = plt.subplots(1,1, figsize=(10, 10))
plot_contours(ax, clf_predict, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

plt.show()
