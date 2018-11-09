import pandas as pd

import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt

from plotly.offline import plot
import plotly
import plotly.graph_objs as go

import warnings

warnings.simplefilter('ignore')

df = pd.read_csv('data/titanic.csv')
df = df.dropna()

df['Survived'] = df.Survived.astype('bool')

print(df.info())

# --- pandas plot ---
fig, axes = plt.subplots(nrows=2, ncols=2)

# распределение пассажиров по возрасту и стоимости билета
p = df.hist(column=['Age', 'Fare'], bins=10, grid=False, ax=axes[0][:])
# по полу
df.groupby('Sex')['Ticket'].count().plot(kind='bar', ax=axes[1, 0], rot=0)
# погибший и спавшийся
df.groupby('Survived')['Ticket'].count().plot(kind='bar', ax=axes[1, 1], rot=0)

# --- seaborn plots ---
sns.pairplot(df[['Age', 'Pclass', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin']])

# плотность вероятности погибнуть (молодых было мало, поэтому и вероятность ниже!)
plt.figure()
p = sns.distplot(df[df.Survived==False].Age)
p.set(xlim=(0, None))

sns.jointplot(df.Pclass, df.Age)

# распределение стоимости среди самых встречаемых возрастов
plt.figure()
top_ages = df.Age.value_counts().sort_values(ascending = False).head(5).index.values
sns.boxplot(y='Age', x='Fare', data=df[df.Age.isin(top_ages)], orient='h')

# тепловая диаграмма родителей + детей относительно возраста и класса
age_class_parch = df.pivot_table(
        index = 'Embarked',
        columns = 'Pclass',
        values = 'Parch',
        aggfunc = sum).fillna(0).applymap(int)
sns.heatmap(age_class_parch, annot=True, fmt='d', linewidths=.5)

# --- plotly plots ---

# зависимость количества родственников от класса
class_df = df.groupby('Pclass')[['SibSp']].sum().join(
    df.groupby('Pclass')[['Parch']].sum()
)
#class_df.columns = ['', '']

trace0 = go.Scatter(
    x=class_df.index,
    y=class_df.SibSp,
    name='siblings+spouses'
)

trace1 = go.Scatter(
    x=class_df.index,
    y=class_df.Parch,
    name='parents+children'
)

data = [trace0, trace1]
layout = {'title': 'Statistics of classes'}

fig = go.Figure(data=data, layout=layout)
plot(fig, show_link=False)

# сравнение количества проданных билетов с прибылью относительно порта
platforms_df = df.groupby('Embarked')[['Fare']].sum().join(
    df.groupby('Embarked')[['Ticket']].count()
)

trace0 = go.Bar(
    x=platforms_df.index,
    y=platforms_df.Fare,
    name='Fare'
)

trace1 = go.Bar(
    x=platforms_df.index,
    y=platforms_df.Ticket,
    name='Tickets'
)

data = [trace0, trace1]
layout = {'xaxis': {'title': 'Embarkation'}}

fig = go.Figure(data=data, layout=layout)
plot(fig, show_link=False)

# распределение возрастов относительно порта
data = []
for em in df.Embarked.unique():
    data.append(
        go.Box(y=df[df.Embarked==em].Age, name=em)
    )

plot(data, show_link = False)

plt.show()
