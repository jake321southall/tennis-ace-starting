import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
tennis_df = pd.read_csv('tennis_stats.csv')
print(tennis_df.head())
print(tennis_df.info())



#EDA:
#Finding the features with the highest correlations
corr_df = tennis_df.corr().abs()
#print(corr_df)
#print(corr_df.unstack())
#print(corr_df.unstack().sort_values(kind="quicksort")[:-len(tennis_df.columns)])


#This one is pretty obvious
sns.scatterplot(y=tennis_df.ServiceGamesPlayed , x=tennis_df.ReturnGamesPlayed, alpha=0.3)
#plt.show()
#plt.clf()

sns.scatterplot(y=tennis_df.BreakPointsOpportunities, x=tennis_df.ReturnGamesPlayed, alpha=0.3)
#plt.show()
#plt.clf()

sns.scatterplot(y=tennis_df.ReturnGamesPlayed, x=tennis_df.BreakPointsOpportunities, alpha=0.3)
#plt.show()
plt.clf()

#Finding the features with the highest correlations to winnings
corr_df_winnings = corr_df['Winnings']
print(corr_df_winnings)
#Wins and games played is pretty obvious, but interestingly there is 0.9 correlation between winnings and 
#break point oppurtunities

sns.scatterplot(y=tennis_df.Winnings, x=tennis_df.BreakPointsOpportunities, alpha=0.3)
plt.show()
plt.clf()

#creating a linear regression model with single feature
feature = tennis_df[['BreakPointsOpportunities']]
target = tennis_df[['Winnings']]

#Creating train and test sets
feature_train, feature_test, target_train, target_test = train_test_split(feature, target, train_size=0.8)

#instantiating the model and fitting to the training sets
model = LinearRegression()
model.fit(feature_train, target_train)

#computing the score on the test sets
score = model.score(feature_test, target_test)
print(f'The score of this break points opportunities model is: {score}.')

#lets visulise this
predictions = model.predict(feature_test)
plt.scatter(target_test,predictions, alpha=0.3)
plt.xlabel('actual winnings')
plt.ylabel('predicted winnings')
plt.title('prediction winnings vs actual winnings using break points opportunities')
plt.show() 
plt.clf()

#Let's do this again for one more feature
feature2 = tennis_df[['Aces']]
target2 = tennis_df[['Winnings']]

#Creating train and test sets
feature_train2, feature_test2, target_train2, target_test2 = train_test_split(feature2, target2, train_size=0.8)

#instantiating the model and fitting to the training sets
model = LinearRegression()
model.fit(feature_train2, target_train2)

#computing the score on the test sets
score = model.score(feature_test2, target_test2)
print(f'The score of this Aces model is: {score}.')

#lets visulise this
predictions2 = model.predict(feature_test2)
plt.scatter(target_test2 ,predictions2 , alpha=0.3)
plt.xlabel('actual winnings')
plt.ylabel('predicted winnings')
plt.title('prediction winnings vs actual winnings using Aces')
plt.show() 
plt.clf()


print(f'Let\'s see if we can improve on these scores by using Aces and Break Points Opportunities to predict winnings')


feature3 = tennis_df[['Aces', 'BreakPointsOpportunities']]
target3 = tennis_df[['Winnings']]

#Creating train and test sets
feature_train3, feature_test3, target_train3, target_test3 = train_test_split(feature3, target3, train_size=0.8)

#instantiating the model and fitting to the training sets
model = LinearRegression()
model.fit(feature_train3, target_train3)

#computing the score on the test sets
score = model.score(feature_test3, target_test3)
print(f'The score of this Aces and BreakPointsOpportunities model is: {score}.')

#lets visulise this
predictions3 = model.predict(feature_test3)
plt.scatter(target_test3 ,predictions3 , alpha=0.3)
plt.xlabel('actual winnings')
plt.ylabel('predicted winnings')
plt.title('prediction winnings vs actual winnings using Aces and BreakPointsOpportunities')
plt.show() 
plt.clf()

