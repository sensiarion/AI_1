import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import warnings

warnings.simplefilter('ignore')

train = pd.read_csv('../../../../data/titanic_train.csv')
print('Dataset information')
train.info();

print('\nДоля выживших:')
print(train['Survived'].mean())
print('\nСреднее значение каждого атрибута, для мужчин:')
print(train[train['Sex'] == 'male'].mean())
print('\nДоля выживших женщин:')
print(train[train['Sex'] == 'female']['Survived'].mean())
print('\nДоля выживших молодых:')
print(train[(train['Age'] > 18) & (train['Age'] < 25)]['Survived'].mean())
#tmp = train['Sex'] == 'male'
#print(type(tmp))
#print(tmp.head())

print('\nЛокализация по пяти первым строкам и атрибутам:')
print(train.loc[0:5, 'Survived':'Age'])
print('\nЛокализация, где используются индексы для атрибутов:')
print(train.iloc[0:5, 0:3])

print('\nПоследний объект фрейма:')
print(train[-1:])

#print('\nМаксимальное значение в каждом столбце:')
#print(train.apply(np.max)) #axis = 1 for each row

print("\nMaping of survived values:")
d = {1 : 'Yes', 0 : 'No'}
print(train['Survived'].map(d).head())
print("\nReplacing of survived values:")
print(train.replace({'Survived': d}).head())

print("\nСтатистическая сводка:")
col_to_show = ['Survived', 'Age']
print(train.groupby(['Sex'])[col_to_show].describe(percentiles=[.5]))
print(train.groupby(['Sex'])[col_to_show].agg([np.mean, np.std, np.min, np.max]))

print("\nТаблица сопряженности:")
print(pd.crosstab(train['Pclass'], train['Sex']))
print(pd.crosstab(train['Pclass'], train['Sex'], normalize=True))

print("\nСводная таблица:")
#value, index, agg
print(train.pivot_table(['Fare', 'Age', 'Pclass'], ['Sex'], aggfunc='mean').head(10))

print("\nВставляем колонку с общим числом родственников:")
total_relatives = train['SibSp'] + train['Parch']
train.insert(len(train.columns), 'Trelatives', total_relatives)
#train.Trelatives = train['SibSp'] + train['Parch'] # doesn't allow
print(train.head(3))

#train.drop([1, 2])
#train.drop([Sex, Survived], axis=1)

#pd.crosstab(train['Pclass'], train['Sex'], margins = True) # margins для строки all (total)

###

print(train.shape)

print("\nРаспределение по классам билетов:")
print(train['Pclass'].value_counts())

train['Pclass'].value_counts().plot(kind='bar', label='Pclass')
#plt.legend()
plt.title('Распределение по классам билетов')

# --- количественные признаки  ---
plt.figure()
features = ['Age', 'SibSp', 'Parch', 'Fare']
corr_matrix = train[features].corr() # корреляция
#print(corr_matrix)
#print(type(corr_matrix)) # dataframe
sns.heatmap(corr_matrix)

train[features].hist() # колич. распред.

sns.pairplot(train.dropna()[['Age', 'SibSp', 'Parch', 'Fare']]) # рассеяние колич признаков

# статистики колич. признаков в группах по выживанию
fig, axes = plt.subplots(nrows=2, ncols=2)
for idx, feat in enumerate(features):
    row = math.floor(idx / 2)
    col = idx % 2
    sns.boxplot(x='Survived', y=feat, data=train, ax=axes[row, col])
    axes[row, col].set_xlabel('Survived')
    axes[row, col].set_ylabel(feat)

_, axes = plt.subplots(1, 2)
sns.boxplot(x='Survived', y='Fare', data=train, ax=axes[0])
sns.violinplot(x='Survived', y='Fare', data=train, ax=axes[1]) # скрипка

plt.figure()
sns.countplot(x='Pclass', hue='Survived', data=train)

plt.figure()
sns.countplot(x='Sex', hue='Survived', data=train)

plt.figure()
sns.countplot(x='Embarked', hue='Survived', data=train)

###

#from sklearn.manifold import TSNE
#from sklearn.preprocessing import StandardScaler
#
#X = train.drop(['PassengerId', 'Cabin', 'Ticket', 'Name'], axis=1)
#X.Sex = pd.factorize(X.Sex)[0]
#X.Embarked = pd.factorize(X.Embarked)[0]
#X = X.dropna()
#
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)
#
#tsne = TSNE(random_state=17)
#tsne_representation = tsne.fit_transform(X_scaled)
#
#plt.figure()
#plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1], c=train['Survived'].map({0: 'blue', 1: 'orange'}));

plt.show()
