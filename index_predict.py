import numpy as np
import tushare as ts
import codecs
import time
from ml_model import MLModel
import stock_predict

# 导入金融数据
with codecs.open('token.txt', 'rb', 'utf-8') as f:
    token = f.read()
ts.set_token(token)
pro = ts.pro_api()

"""
generate_x_and_y方法的调参
对于中证500 '000905.SH'，small_num=0.4 , big_num=1.2
对于沪深300 '399300.SZ'，small_num=0.2 , big_num=0.72
"""

stock = '399300.SZ'  # 399300.SZ   000905.SH
divide_date = '20190101'

quotes = pro.index_daily(ts_code=stock, start_date='20090101')

print("开始 :", stock, ", 时间 :", time.ctime())
quotes = quotes.sort_values(by=['trade_date'], ascending=True)
train_quotes = quotes[quotes['trade_date'] < divide_date].sort_values(by=['trade_date'], ascending=True)

time_steps = 5  # 每次训练的数据长度

total_days = len(quotes)
train_days = len(train_quotes)

# 数据
close_total = np.array(quotes['close'])
open_total = np.array(quotes['open'])
high_total = np.array(quotes['high'])
low_total = np.array(quotes['low'])
change_total = np.array(quotes['pct_chg'])
turnover_vol_total = np.array(quotes['vol'])
turnover_value_total = np.array(quotes['amount'])
mean_total = (np.array(quotes['high']) + np.array(quotes['low'])) / 2

x_total, y_total = stock_predict.generate_x_and_y(time_steps, change_total,
                                                  change_total, open_total, close_total, high_total, low_total, turnover_value_total)

x_train = x_total[0:train_days - time_steps]
y_train = y_total[0:train_days - time_steps]
x_test = x_total[train_days - time_steps:]
y_test = y_total[train_days - time_steps:]
y_test = np.argmax(y_test, axis=1) - 2

# 构建模型
model = MLModel(input_shape=(x_train.shape[1], x_train.shape[2]), stock_name=stock)

# 训练模型
model.train_model(x_train=x_train, y_train=y_train, epoch=5, batch_size=32)

# 预测结果
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1) - 2
print("y_predict: ", y_predict)

correct_num = 0
rough_correct_num = 0
for i in range(y_test.shape[0]):
    if y_predict[i] == y_test[i]:
        correct_num += 1
    if (y_predict[i] > 0 and y_test[i] > 0) or (y_predict[i] < 0 and y_test[i] < 0) or (
            y_predict[i] == 0 and y_test[i] == 0):
        rough_correct_num += 1
print(stock, "准确的大涨、大跌、震荡、小跌、大跌的预测准确率：", correct_num / y_test.shape[0])
print(stock, "涨跌平的预测准确率：", rough_correct_num / y_test.shape[0])

stock_predict.stock_operation(stock_name=stock,
                              change=change_total[train_days:],
                              close=close_total[train_days - 1:],
                              mean=mean_total[train_days:],
                              predict_state=y_predict)

print("结束 :", stock, ", =====================================时间 :", time.ctime())
