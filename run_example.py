import numpy as np
import matplotlib.pyplot as plt
import changetree
import random
import pandas as pd


random.seed(0)
mean = [0, 0] # mean of first 50 points
mean2 = [10, -10] # mean of last 50 points
cov = np.identity(2)

ntrees = 10
w = 1
lim = 10

"""
# Generate Gaussian data with change point at t=51 (or t=50 under 0-indexing), with Gaussian means (0,0) and (10,-10) resp. 
x, y = np.random.multivariate_normal(mean, cov, n//2).T
x2, y2 = np.random.multivariate_normal(mean2, cov, n//2).T
#X = np.concatenate([np.array([x, y]).T, np.array([x2, y2]).T])
"""

df= pd.read_csv('data/series_order.csv')
X= df.iloc[:, :].values
n = X.shape[0]

print('OFFLINE ALGORITHM:')
score, changes = changetree.change_score(X, ntrees, w, lim)
plt.figure()
plt.plot(range(n), score, 'r-x')
plt.axvline(x=changes[0])
plt.xlabel('Time tick')
plt.ylabel('Scores (offline)')
plt.show()


print('ONLINE ALGORITHM:')
score_online, changes_online = changetree.change_score_online(X, ntrees, w, lim)
plt.figure()
plt.plot(range(n), score_online, 'r-x')
plt.axvline(x=changes_online[0])
plt.xlabel('Time tick')
plt.ylabel('Scores (online)')
plt.show()


df_score= pd.DataFrame({"score": score})
dates = pd.date_range('1/1/2000', periods= X.shape[0])
df_score.set_index([dates])


plt.figure(figsize=(12, 6))
plt.plot(dates, df_score.score, 'r-x')
plt.axvline(x=changes[0])

