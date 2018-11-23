import pandas as pd
import numpy as np 
#pd.set_option('display.max_columns', 20)
#pd.set_option('display.max_rows', 5)

df = pd.read_csv('data/telecom_churn.csv')

d = {'No' : False, 'Yes' : True}
df['International plan'] = df['International plan'].map(d)

df['Churn'] = df['Churn'].astype('int64')

#print(df.shape)
print(df.columns)
print(df.info())

print(df.head())

print(df.sort_values(by='Total day charge', ascending=False).head())

print(df.apply(np.max))

#
print()

import warnings
warnings.simplefilter('ignore')

import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/video_games_sales.csv')
df.info()

#df = df.replace('N/A', np.NaN)
#df['User_Score'] = df.User_Score.replace('tbd', np.nan)
df = df.dropna()
print(df.shape)

df['User_Score'] = df.User_Score.astype('float64')
df['Year_of_Release'] = df.Year_of_Release.astype('int64')
df['User_Count'] = df.User_Count.astype('int64')
df['Critic_Count'] = df.Critic_Count.astype('int64')

useful_cols = ['Name', 'Platform', 'Year_of_Release', 'Genre', 'Global_Sales', 'Critic_Count', 'Critic_Count', 'User_Score', 'User_Count', 'Rating']
print(df[useful_cols].head())

sales_df = df[[x for x in df.columns if 'Sales' in x ] + ['Year_of_Release']]
sales_df.groupby('Year_of_Release').sum().plot()
#sales_df.groupby('Year_of_Release').sum().plot(kind='bar', rot=45)

cols = ['Global_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']
sns_plot = sns.pairplot(df[cols])
#sns_plot.savefig('output/pairplot.png')

plt.figure('1')
sns.distplot(df.Critic_Score)

sns.jointplot(df.Critic_Score, df.User_Score)

plt.figure('2')
#print(df.Platform)
#print(df.Platform.value_counts())
top_platforms = df.Platform.value_counts().sort_values(ascending = False).head(5).index.values
sns.boxplot(y='Platform', x='Critic_Score', data=df[df.Platform.isin(top_platforms)], orient='h')

platform_genre_sales = df.pivot_table(
        index = 'Platform',
        columns = 'Genre',
        values = 'Global_Sales',
        aggfunc = sum).fillna(0).applymap(float)
sns.heatmap(platform_genre_sales, annot=True, fmt='.1f', linewidths=.5)

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go

#init_notebook_mode(connected=True)

years_df = df.groupby('Year_of_Release')[['Global_Sales']].sum().join(
    df.groupby('Year_of_Release')[['Name']].count()
)
years_df.columns = ['Global_Sales', 'Number_of_Games']

trace0 = go.Scatter(
    x=years_df.index,
    y=years_df.Global_Sales,
    name='Global Sales'
)

trace1 = go.Scatter(
    x=years_df.index,
    y=years_df.Number_of_Games,
    name='Number of games released'
)

data = [trace0, trace1]
layout = {'title': 'Statistics of video games'}

fig = go.Figure(data=data, layout=layout)
#iplot(fig, show_link=False)
plot(fig, show_link=False)

#

platforms_df = df.groupby('Platform')[['Global_Sales']].sum().join(
    df.groupby('Platform')[['Name']].count()
)
platforms_df.columns = ['Global_Sales', 'Number_of_Games']
platforms_df.sort_values('Global_Sales', ascending=False, inplace=True)

trace0 = go.Bar(
    x=platforms_df.index,
    y=platforms_df.Global_Sales,
    name='Global Sales'
)

trace1 = go.Bar(
    x=platforms_df.index,
    y=platforms_df.Number_of_Games,
    name='Number of games released'
)

data = [trace0, trace1]
layout = {'title': 'Share of platforms', 'xaxis': {'title': 'platform'}}

fig = go.Figure(data=data, layout=layout)
plot(fig, show_link=False, )

#

data = []
for genre in df.Genre.unique():
    data.append(
        go.Box(y=df[df.Genre==genre].Critic_Score, name=genre)
    )

plot(data, show_link = False)

plt.show()
