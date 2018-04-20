import numpy as np
import rbm
import projectLib as lib
import matplotlib.pyplot as plt
import math

training = lib.getTrainingData()
validation = lib.getValidationData()
# You could also try with the chapter 4 data
# training = lib.getChapter4Data()

trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)

K = 5

# SET PARAMETERS HERE!!!
# number of hidden units
# F = 11
F = 10
rmse = []
epochs = 40
gradientLearningRate = 0.1
# for regularization
# still finetuning param
# ld = 0.15
mu = 0.95
alpha = 0.05

# Initialise all our arrays

W = rbm.getInitialWeights(trStats["n_movies"], F, K)
posprods = np.zeros(W.shape)
negprods = np.zeros(W.shape)
grad = np.zeros(W.shape)

for epoch in range(1, epochs):
        # in each epoch, we'll visit all users in a random order
        visitingOrder = np.array(trStats["u_users"])
        np.random.shuffle(visitingOrder)
        ###############################################
        # Adaptive Learning Rates
        # Adjust learning rate such that it decreases with
        # num of epochs
        # adaptive_rate = alpha / math.sqrt(epoch)
        ###############################################

        ###############################################
        # Nestorov Momentum - stores previous gradient
        # Resets current gradient after each epoch
        # prev_grad = grad
        # grad = np.zeros(W.shape)
        ###############################################
        for user in visitingOrder:
                # get the ratings of that user
                ratingsForUser = lib.getRatingsForUser(user, training)

                # build the visible input
                v = rbm.getV(ratingsForUser)

                # get the weights associated to movies the user has seen
                weightsForUser = W[ratingsForUser[:, 0], :, :]

                ### LEARNING ###
                # propagate visible input to hidden units
                posHiddenProb = rbm.visibleToHiddenVec(v, weightsForUser)
                # get positive gradient
                # note that we only update the movies that this user has seen!
                posprods[ratingsForUser[:, 0], :, :] += rbm.probProduct(v, posHiddenProb)

                ### UNLEARNING ###
                # sample from hidden distribution
                sampledHidden = rbm.sample(posHiddenProb)
                # propagate back to get "negative data"
                negData = rbm.hiddenToVisible(sampledHidden, weightsForUser)
                # propagate negative data to hidden units
                negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser)
                # get negative gradient
                # note that we only update the movies that this user has seen!
                negprods[ratingsForUser[:, 0], :, :] += rbm.probProduct(negData, negHiddenProb)
	            
                # we average over the number of users in the batch (if we use mini-batch)
                # without adaptive lr
                grad += gradientLearningRate * (posprods - negprods)
                ###############################################
                # Adaptive learning rate Implementation
                # grad += adaptive_rate * (posprods - negprods)
                ###############################################
                #REGULARIZATION#
                #Not working yet#
                # grad += gradientLearningRate * ((posprods - negprods) - (ld*W*0.5))
                
                # without momentum
                W += grad
                ###############################################
                # Nestorov Momentum implementation
                # W += mu * (grad + mu * prev_grad) + gradientLearningRate * (posprods - negprods)
                ###############################################
                ###############################################
                # Nestorov Momentum with adaptive learning rate
                # W += mu * (grad + mu * prev_grad) + adaptive_rate * (posprods - negprods)
                ###############################################

        # Print the current RMSE for training and validation sets
        # this allows you to control for overfitting e.g
        # We predict over the training set

        tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, training)
        trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)

        # We predict over the validation set
        vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, training)
        vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)

        print("### EPOCH %d ###" % epoch)
        print("Training loss = %f" % trRMSE)
        print("Validation loss = %f" % vlRMSE)

        rmse.append(trRMSE)


# Plots RMSE against epochs
x = np.arange(1,epochs)
y = np.asarray(rmse)
# Find min RMSE and at which epoch
print("### MIN RMSE = %f ###" % y[np.argmin(y)])
print("### AT EPOCH NUM = %d ###" % np.argmin(y)+1)
plt.plot(x,y)
plt.show()

### END ###
# This part you can write on your own
# you could plot the evolution of the training and validation RMSEs for example

# predictedRatings = np.array([rbm.predictForUser(user, W, training) for user in trStats["u_users"]])
# np.savetxt("predictedRatings.txt", predictedRatings)
