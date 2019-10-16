import numpy as np
import tushare as ts
import codecs
import time
from sklearn.preprocessing import scale
from keras.utils import np_utils
import matplotlib.pyplot as plt
from ml_model import MLModel

# 导入金融数据
with codecs.open('token.txt', 'rb', 'utf-8') as f:
    token = f.read()
ts.set_token(token)
pro = ts.pro_api()

stock = '399300.SZ'  # 399300.SZ   000905.SH

quotes = pro.index_daily(ts_code=stock, start_date='20180101')

print("开始 :", stock, ", 时间 :", time.ctime())
quotes = quotes.sort_values(by=['trade_date'], ascending=True)

window_len = 20  # 窗口的长度
time_steps = 2  # 每次训练的数据长度

# 数据
close_total = np.array(quotes['close'])
open_total = np.array(quotes['open'])
high_total = np.array(quotes['high'])
low_total = np.array(quotes['low'])
change_total = np.array(quotes['pct_chg'])
turnover_vol_total = np.array(quotes['vol'])
turnover_value_total = np.array(quotes['amount'])
mean_total = (np.array(quotes['high']) + np.array(quotes['low'])) / 2

base_money = close_total[window_len-1]  # 每天的金额，以第一天的前一天的收盘价为基础金额
base_money_fee = base_money  # 每天的金额，含手续费计算
base = 1  # 用来计算收益率
base_fee = base  # 用来计算含手续费的收益率
model_line = [base_money]  # 记录每天的金额的列表
model_line_fee = [base_money]  # 记录含手续费的每天的金额的列表
print(stock, '初始金额 : ', base_money, ' , 实际最终金额 : ', close_total[-1])

buyed = 1  # 股票是否已经购买的状态
buy_num = 0  # 购买股票的天数
hold_num = 0  # 持有股票的天数
sell_num = 0  # 抛出股票的天数
empty_num = 0  # 空仓的天数

up_num = 0  # 预测涨正确的天数
down_num = 0  # 预测跌正确的天数
medium_num = 0  # 预测平正确的天数

for i in range(change.shape[0]):  # 从第一天起到最后一天，计算金额变化

    if predict_state[i] > 0 and change[i] > 1:
        up_num += 1
    elif predict_state[i] < 0 and change[i] < -1:
        down_num += 1
    elif predict_state[i] == 0 and -1 <= change[i] <= 1:
        medium_num += 1

    if predict_state[i] > 0 and (not buyed):  # 预测结果涨 且 没有持有股票， 买入，手续费0.00032
        buyed = 1
        buy_num += 1
        rate_temp = (mean[i] - close[i]) / close[i]  # 基于第二天股票均价相对于第一天收盘价的涨跌幅
        base = base * (1 + rate_temp)
        base_money = base_money * (1 + rate_temp)
        base_fee = base_fee * (1 + rate_temp) * (1 - 0.00032)
        base_money_fee = base_money_fee * (1 + rate_temp) * (1 - 0.00032)
    elif predict_state[i] < 0 and buyed:  # 预测结果为平或跌 且 持有股票，抛出，手续费0.00132
        buyed = 0
        sell_num += 1
        rate_temp = (mean[i] - close[i]) / close[i]  # 基于第二天股票均价相对于第一天收盘价的涨跌幅
        base = base * (1 + rate_temp)
        base_money = base_money * (1 + rate_temp)
        base_fee = base_fee * (1 + rate_temp) * (1 - 0.00132)
        base_money_fee = base_money_fee * (1 + rate_temp) * (1 - 0.00132)
    elif predict_state[i] >= 0 and buyed:  # 预测结果为涨 且 持有股票，不进行操作
        hold_num += 1
        base = base * (1 + change[i] / 100)
        base_money = base_money * (1 + change[i] / 100)
        base_fee = base_fee * (1 + change[i] / 100)
        base_money_fee = base_money_fee * (1 + change[i] / 100)
    else:  # 预测结果为跌 且 没有持有股票，不进行操作
        empty_num += 1

    model_line.append(base_money)
    model_line_fee.append(base_money_fee)

model_line = np.array(model_line)
model_line_fee = np.array(model_line_fee)

print(stock_name, '预测涨正确的天数', up_num)
print(stock_name, '预测跌正确的天数', down_num)
print(stock_name, '预测平正确的天数', medium_num)

print(stock_name, '总天数:', change.shape[0], ',买入天数: ', buy_num, ',卖出天数: ', sell_num, ',持有天数: ',
      hold_num, ',空仓天数: ', empty_num)
print(stock_name, '最终金额: ', base_money, ' , 最终金额(含交易费): ', base_money_fee)
print(stock_name, "收益率: ", base - 1, " , 收益率(含交易费): ", base_fee - 1)

# 绘制曲线图
fig = plt.figure()
plt.plot(close, color='green', label='Real Stock Price')
plt.plot(model_line, color='red', label='Predicted Without Fee')
plt.plot(model_line_fee, color='yellow', label='Predicted With Fee')
plt.title(label='Stock Prediction')
plt.xlabel(xlabel='Time')
plt.ylabel(ylabel='Stock Price')
plt.legend(loc='upper left')
plt.show()
fig.savefig("pictures\\lines_" + stock_name + ".png")


def generate_x_and_y_3(time_steps, change, *xs):
    flat_num = 0.2  # (-flat_num, flat_num) 平

    data_num = 0  # 参数的个数

    if len(xs) < 1:
        print('error: 没有传入参数XS！！！')
        return
    x_len = len(change)
    x_list = []
    for x_temp in xs:
        if len(x_temp) != x_len:
            print('error: XS参数长度不相等！！！')
            return
        x_list.append(x_temp)
        data_num += 1

    x_unscaled = np.column_stack(x_list)
    x_scaled = scale(X=x_unscaled, axis=0)  # 归一化

    x = []
    y = []
    for i in range(time_steps, x_scaled.shape[0]):
        x.append(x_scaled[i - time_steps: i])
        y_change = change[i]
        if y_change < -flat_num:
            y.append(0)
        elif -flat_num <= y_change <= flat_num:
            y.append(1)
        elif y_change > flat_num:
            y.append(2)
    # 将数据转化为数组
    x, y = np.array(x), np.array(y)
    # 因为LSTM要求输入的数据格式为三维的，[training_number, time_steps, data_num]，因此对数据进行相应转化
    x = np.reshape(x, (x.shape[0], x.shape[1], data_num))
    y = np_utils.to_categorical(y, num_classes=3)

    return x, y
