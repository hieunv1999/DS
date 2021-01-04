from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, norm, skewtest, kurtosistest
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf


class CrawlStock:
    def __init__(self, url):
        self.column = []
        self.date = []
        self.Open = []
        self.High = []
        self.Low = []
        self.Close = []
        self.Adj_Close = []
        self.Volume = []
        self.url = url
        self.page = urlopen(url)
        self.soup = BeautifulSoup(self.page, 'html.parser')
        self.df = None

    def crawl_title(self):
        title_html = self.soup.find_all('tr', {'class': 'C($tertiaryColor) Fz(xs) Ta(end)'})
        for title_con in title_html:
            title = title_con.find_all('th')
            for name in title:
                self.column.append(name.text)

    def crawl_price(self):
        price_stock = self.soup.find_all('tr', {'class': 'BdT Bdc($seperatorColor) Ta(end) Fz(s) Whs(nw)'})
        for price_day in price_stock:
            id_column = price_day.find_all('td')
            if len(id_column) == 7:
                self.date.append(id_column[0].text)
                self.Open.append(id_column[1].text)
                self.High.append(id_column[2].text)
                self.Low.append(id_column[3].text)
                self.Close.append(id_column[4].text)
                self.Adj_Close.append(id_column[5].text)
                self.Volume.append(id_column[6].text)

    def tranfer_dataframe(self):
        data = {
            'Date': self.date,
            'Open': self.Open,
            'High': self.High,
            'Low': self.Low,
            'Close': self.Close,
            'Adj Close': self.Adj_Close,
            'Volume': self.Volume
        }
        self.df = pd.DataFrame(data=data)


company = 'VXRT'
url_stock = 'https://finance.yahoo.com/quote/' + company + '/history?p=' + company
price = CrawlStock(url_stock)
price.crawl_title()
price.crawl_price()
price.tranfer_dataframe()
returns = []
x = price.df['Close'].tolist()
print(type(x))
for i in range(1,len(x)):
    y = float(x[i])-float(x[i-1])
    y = y/float(x[i-1])
    returns.append(y)
plt.hist(returns,bins="rice",label="Daily close price")
plt.legend()
plt.show()

plt.boxplot(returns,labels=["Daily close price"])
plt.show()

plt.plot(returns)
plt.xlabel("Time")
plt.ylabel("Daily returns")
plt.show()

plot_pacf(returns,lags=20)
plt.show()

