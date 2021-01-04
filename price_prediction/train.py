from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from preprocessing import PreProcess
from model import LSTM_4
import numpy
import matplotlib.pyplot as plt
import math
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = PreProcess()
dataset.crawl_stock()
dataset.creat_traindata()
trainX = dataset.trainX
trainY = dataset.trainY
testX = dataset.testX
testY = dataset.testY
look_back = 1
model = LSTM_4(trainX, trainY)
model.create_model()
model.train()
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainPredict , trainY = dataset.inverse_data(trainPredict,trainY)
testPredict, testY = dataset.inverse_data(testPredict,testY)
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % trainScore)
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % testScore)
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset.dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset.dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset.dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(dataset.data)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()