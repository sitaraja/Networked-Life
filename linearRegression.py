import numpy as np
import projectLib as lib

# shape is movie,user,rating
training = lib.getTrainingData()
validation = lib.getValidationData()

#some useful stats
trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)
rBar = np.mean(trStats["ratings"])

# we get the A matrix from the training dataset
def getA(training):
    A = np.zeros((trStats["n_ratings"], trStats["n_movies"] + trStats["n_users"]))
    index = 0
    for row in training:
        if row[len(row)-1] > 0:
            A[index][row[0]] = 1
            A[index][trStats["n_movies"] + row[1]] = 1
        index += 1
    return A

# we also get c
def getc(rBar, ratings):
    c = []
    for rating in ratings:
        c.append(rating - rBar)
    return np.transpose(np.asarray(c))

# apply the functions
A = getA(training)
c = getc(rBar, trStats["ratings"])

# compute the estimator b
def param(A, c):
    x = np.linalg.inv(np.dot(np.transpose(A),A))
    y = np.dot(np.transpose(A),c)
    return np.dot(x,y)

# compute the estimator b with a regularisation parameter l
# note: lambda is a Python keyword to define inline functions
#       so avoid using it as a variable name!
def param_reg(A, c, l):
    x = np.linalg.inv(np.dot(np.transpose(A),A) +  l * np.identity(np.shape(A)[1]))
    y = np.dot(np.transpose(A),c)
    return np.dot(x,y)

# from b predict the ratings for the (movies, users) pair
def predict(movies, users, rBar, b):
    n_predict = len(users)
    p = np.zeros(n_predict)
    for i in range(0, n_predict):
        rating = rBar + b[movies[i]] + b[trStats["n_movies"] + users[i]]
        if rating > 5: rating = 5.0
        if rating < 1: rating = 1.0
        p[i] = rating
    return p

# Unregularised version (<=> regularised version with l = 0)
# b = param(A, c)

# Regularised version
l = 1
b = param_reg(A, c, l)

print "Linear regression, l = %f" % l
print "RMSE for training %f" % lib.rmse(predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"])
print "RMSE for validation %f" % lib.rmse(predict(vlStats["movies"], vlStats["users"], rBar, b), vlStats["ratings"])
# Linear regression, l =1.0
# RMSE for training 0.847741
# RMSE for validation 1.060366