import pandas as pd
# Importing all required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

media = pd.read_csv('Data/mediacompany.csv')

# Remove un-necessary  column
media = media.drop('Unnamed: 7', axis=1)

# Convert Date column into pandas Datetime type
media['Date'] = pd.to_datetime(media['Date'])
# print(media['Date'])


# Deriving "days since the show started"
d1 = media['Date'].dt.day
media['Day'] = d1
# print(media.head(5))

# #days vs Views_show in line chart
# media.plot.line(x='Day', y='Views_show')
# plt.show()

# Scatter Plot (days vs Views_show)
# colors = (0, 0, 0)
# plt.scatter(media.Day, media.Views_show, c=colors, alpha=1)
# plt.title('Scatter plot pythonspot.com')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()
#
#
# # # plot for days vs Views_show and days vs Ad_impressions
# #
# fig = plt.figure()
# host = fig.add_subplot(111)
#
# par1 = host.twinx()
# par2 = host.twinx()
#
# host.set_xlabel("Day")
# host.set_ylabel("View_Show")
# par1.set_ylabel("Ad_impression")
#
# color1 = plt.cm.viridis(0)
# color2 = plt.cm.viridis(0.5)
# color3 = plt.cm.viridis(.9)
#
# p1, = host.plot(media.Day, media.Views_show, color=color1, label="View_Show")
# p2, = par1.plot(media.Day, media.Ad_impression, color=color2, label="Ad_impression")
#
# lns = [p1, p2]
# host.legend(handles=lns, loc='best')
#
# # right, left, top, bottom
# par2.spines['right'].set_position(('outward', 60))
# # no x-ticks
# par2.xaxis.set_ticks([])
# # Sometimes handy, same for xaxis
# # par2.yaxis.set_ticks_position('right')
#
# host.yaxis.label.set_color(p1.get_color())
# par1.yaxis.label.set_color(p2.get_color())
# plt.show()
# plt.savefig("pyplot_multiple_y-axis.png", bbox_inches='tight')
#
# # Derived Metrics
# Weekdays are taken such that 1 corresponds to Sunday and 7 to Saturday
# Generate the weekday variable
media['weekday'] = (media['Day'] + 3) % 7
media.weekday.replace(0, 7, inplace=True)
media['weekday'] = media['weekday'].astype(int)
# print(media.head())


# # Putting feature variable to X and Putting response variable to y
# X = media[['Visitors', 'weekday']]
# y = media['Views_show']
#
# from sklearn.linear_model import LinearRegression
# lm = LinearRegression()
# # fit the model to the training data
# lm.fit(X, y)
#
# import statsmodels.api as sm
# # Unlike SKLearn, statsmodels don't automatically fit a constant,
# # so you need to use the method sm.add_constant(X) in order to add a constant.
#
# X = sm.add_constant(X)
# # create a fitted model in one line
# lm_1 = sm.OLS(y, X).fit()
# print(lm_1.summary()) # R square is = 0.290

# create Weekend variable, with value 1 at weekends and 0 at weekdays
def cond(i):
    if i % 7 == 5:
        return 1
    elif i % 7 == 4:
        return 1
    else:
        return 0
    return i


media['weekend'] = [cond(i) for i in media['Day']]


# Putting feature variable to X
# X = media[['Visitors', 'weekend']]
#
# # Putting response variable to y
# y = media[['Views_show']]
# import statsmodels.api as sm
#
# X = sm.add_constant(X)
# lm_2 = sm.OLS(y, X).fit()
# print(lm_2.summary())
#
# # Putting feature variable to X
# X = media[['Visitors', 'weekend', 'Character_A']]
# # Putting response variable to y
# y = media['Views_show']
#
# import statsmodels.api as sm
#
# # Unlike SKLearn, statsmodels don't automatically fit a constant,
# # so you need to use the method sm.add_constant(X) in order to add a constant.
# X = sm.add_constant(X)
# # create a fitted model in one line
# lm_3 = sm.OLS(y, X).fit()
# print(lm_3.summary()) # R square is = 0.389


# # Create lag variable
# media['Lag_Views'] = np.roll(media['Views_show'], 1)
# media.Lag_Views.replace(108961, 0, inplace=True)
# media.head(10)
#
# # Putting feature variable to X
# X = media[['Visitors', 'Character_A', 'Lag_Views', 'weekend']]
#
# # Putting response variable to y
# y = media['Views_show']
# import statsmodels.api as sm
#
# # Unlike SKLearn, statsmodels don't automatically fit a constant,
# # so you need to use the method sm.add_constant(X) in order to add a constant.
# X = sm.add_constant(X)
# # create a fitted model in one line
# lm_4 = sm.OLS(y, X).fit()
# print(lm_4.summary())
#
# plt.figure(figsize=(20, 10))  # Size of the figure
# sns.heatmap(media.corr(), annot=True)
#
# # Putting feature variable to X
# X = media[['weekend', 'Character_A', 'Views_platform']]
#
# # Putting response variable to y
# y = media['Views_show']
# import statsmodels.api as sm
#
# # Unlike SKLearn, statsmodels don't automatically fit a constant,
# # so you need to use the method sm.add_constant(X) in order to add a constant.
# X = sm.add_constant(X)
# # create a fitted model in one line
# lm_5 = sm.OLS(y, X).fit()
# print(lm_5.summary())
#
# # Putting feature variable to X
# X = media[['weekend', 'Character_A', 'Visitors']]
#
# # Putting response variable to y
# y = media[['Views_show']]
# import statsmodels.api as sm
#
# # Unlike SKLearn, statsmodels don't automatically fit a constant,
# # so you need to use the method sm.add_constant(X) in order to add a constant.
# X = sm.add_constant(X)
# # create a fitted model in one line
# lm_6 = sm.OLS(y, X).fit()
# print(lm_6.summary())
#
# # Putting feature variable to X
# X = media[['weekend', 'Character_A', 'Visitors', 'Ad_impression']]
#
# # Putting response variable to y
# y = media['Views_show']
# import statsmodels.api as sm
#
# # Unlike SKLearn, statsmodels don't automatically fit a constant,
# # so you need to use the method sm.add_constant(X) in order to add a constant.
# X = sm.add_constant(X)
# # create a fitted model in one line
# lm_7 = sm.OLS(y, X).fit()
# print(lm_7.summary())
#
# # Putting feature variable to X
# X = media[['weekend', 'Character_A', 'Ad_impression']]
#
# # Putting response variable to y
# y = media['Views_show']
# import statsmodels.api as sm
#
# # Unlike SKLearn, statsmodels don't automatically fit a constant,
# # so you need to use the method sm.add_constant(X) in order to add a constant.
# X = sm.add_constant(X)
# # create a fitted model in one line
# lm_8 = sm.OLS(y, X).fit()
# print(lm_8.summary())
#
# # Ad impression in million
# media['ad_impression_million'] = media['Ad_impression'] / 1000000
# # Putting feature variable to X
# X = media[['weekend', 'Character_A', 'ad_impression_million', 'Cricket_match_india']]
#
# # Putting response variable to y
# y = media['Views_show']
# import statsmodels.api as sm
#
# # Unlike SKLearn, statsmodels don't automatically fit a constant,
# # so you need to use the method sm.add_constant(X) in order to add a constant.
# X = sm.add_constant(X)
# # create a fitted model in one line
# lm_9 = sm.OLS(y, X).fit()
# print(lm_9.summary())
#
# # Putting feature variable to X
# X = media[['weekend', 'Character_A', 'ad_impression_million']]
#
# # Putting response variable to y
# y = media['Views_show']
# import statsmodels.api as sm
#
# # Unlike SKLearn, statsmodels don't automatically fit a constant,
# # so you need to use the method sm.add_constant(X) in order to add a constant.
# X = sm.add_constant(X)
# # create a fitted model in one line
# lm_10 = sm.OLS(y, X).fit()
# print(lm_10.summary())
#
# # Making predictions using the model lm_10
# X = media[['weekend', 'Character_A', 'ad_impression_million']]
# X = sm.add_constant(X)
# Predicted_views = lm_10.predict(X)
#
# from sklearn.metrics import mean_squared_error, r2_score
#
# mse = mean_squared_error(media.Views_show, Predicted_views)
# r_squared = r2_score(media.Views_show, Predicted_views)
# print('Mean_Squared_Error :', mse)
# print('r_square_value :', r_squared)
#
# # Actual vs Predicted
# c = [i for i in range(1, 81, 1)]
# fig = plt.figure()
# plt.plot(c, media.Views_show, color="blue", linewidth=2.5, linestyle="-")
# plt.plot(c, Predicted_views, color="red", linewidth=2.5, linestyle="-")
# fig.suptitle('Actual and Predicted', fontsize=20)  # Plot heading
# plt.xlabel('Index', fontsize=18)  # X-label
# plt.ylabel('Views', fontsize=16)  # Y-label
#
# # Error terms
# c = [i for i in range(1, 81, 1)]
# fig = plt.figure()
# plt.plot(c, media.Views_show - Predicted_views, color="blue", linewidth=2.5, linestyle="-")
# fig.suptitle('Error Terms', fontsize=20)  # Plot heading
# plt.xlabel('Index', fontsize=18)  # X-label
# plt.ylabel('Views_show-Predicted_views', fontsize=16)  # Y-label
#
# from sklearn.metrics import mean_squared_error, r2_score
#
# mse = mean_squared_error(media.Views_show, Predicted_views)
# r_squared = r2_score(media.Views_show, Predicted_views)
#
# # Making predictions using the model
# X = media[['weekend', 'Character_A', 'Visitors']]
# X = sm.add_constant(X)
# Predicted_views = lm_6.predict(X)
# print('Mean_Squared_Error :', mse)
# print('r_square_value :', r_squared)
#
# # Actual vs Predicted
# c = [i for i in range(1, 81, 1)]
# fig = plt.figure()
# plt.plot(c, media.Views_show, color="blue", linewidth=2.5, linestyle="-")
# plt.plot(c, Predicted_views, color="red", linewidth=2.5, linestyle="-")
# fig.suptitle('Actual and Predicted', fontsize=20)  # Plot heading
# plt.xlabel('Index', fontsize=18)  # X-label
# plt.ylabel('Views', fontsize=16)  # Y-label
#
# # Error terms
# c = [i for i in range(1, 81, 1)]
# fig = plt.figure()
# plt.plot(c, media.Views_show - Predicted_views, color="blue", linewidth=2.5, linestyle="-")
# fig.suptitle('Error Terms', fontsize=20)  # Plot heading
# plt.xlabel('Index', fontsize=18)  # X-label
# plt.ylabel('Views_show-Predicted_views', fontsize=16)  # Y-label
