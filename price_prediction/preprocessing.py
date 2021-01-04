import numpy
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from crawldata import CrawlStock


class PreProcess:
    def __init__(self):
        self.data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.dataset = None
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None

    def crawl_stock(self):
        company = 'VXRT'
        url_stock = 'https://finance.yahoo.com/quote/' + company + '/history?p=' + company
        price = CrawlStock(url_stock)
        price.crawl_title()
        price.crawl_price()
        price.tranfer_dataframe()
        x = numpy.array(price.df['Close'].values.astype('float32'))
        self.data = x.reshape(-1, 1)
        self.dataset = self.scaler.fit_transform(self.data)

    def create_dataset(self, dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return numpy.array(dataX), numpy.array(dataY)

    def creat_traindata(self):
        look_back = 1
        numpy.random.seed(7)
        train_size = int(len(self.dataset) * 0.67)
        train, test = self.dataset[0:train_size, :], self.dataset[train_size:len(self.dataset), :]
        self.trainX, self.trainY = self.create_dataset(train, look_back)
        self.testX, self.testY = self.create_dataset(test, look_back)
        self.trainX = numpy.reshape(self.trainX, (self.trainX.shape[0], 1, self.trainX.shape[1]))
        self.testX = numpy.reshape(self.testX, (self.testX.shape[0], 1, self.testX.shape[1]))

    def inverse_data(self, trainPredict_, trainY_):
        trainPredict = self.scaler.inverse_transform(trainPredict_)
        trainY = self.scaler.inverse_transform([trainY_])
        return trainPredict, trainY


data = PreProcess()
data.crawl_stock()
data.creat_traindata()
print(data.trainX)
