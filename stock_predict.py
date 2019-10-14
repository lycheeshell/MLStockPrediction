import multiprocessing
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import scale
from keras.utils import np_utils
import matplotlib.pyplot as plt
from ml_model import MLModel


def stock_predict(stock, quotes, divide_date):
    try:
        print("开始 :", stock, ", 时间 :", time.ctime())
        quotes = quotes.sort_values(by=['tradeDate'], ascending=True)
        train_quotes = quotes[quotes['tradeDate'] < divide_date].sort_values(by=['tradeDate'], ascending=True)

        time_steps = 240  # 每次训练的数据长度，一年大概240个交易日

        total_days = len(quotes) - 1
        train_days = len(train_quotes) - 1  # 减去计算不到涨跌幅的第一天

        # 数据
        close_total = np.array(quotes['closePrice'])
        change_total = np.zeros(total_days)  # 涨跌幅
        for i in range(total_days):
            change_total[i] = (close_total[i + 1] - close_total[i]) / close_total[i] * 100
        turnover_vol_total = np.array(quotes['turnoverVol'])
        turnover_value_total = np.array(quotes['turnoverValue']) / 10000
        negmarket_value_total = np.array(quotes['negMarketValue']) / 1000000

        x_total, y_total = generate_x_and_y(time_steps, change_total, change_total, turnover_vol_total[1:],
                                            turnover_value_total[1:],
                                            negmarket_value_total[1:])

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
            if (y_predict[i] > 0 and y_test[i] > 0) or (y_predict[i] < 0 and y_test[i] < 0) or (y_predict[i] == 0 and y_test[i] == 0):
                rough_correct_num += 1
        print(stock, "准确的大涨、大跌、震荡、小跌、大跌的预测准确率：", correct_num / y_test.shape[0])
        print(stock, "涨跌平的预测准确率：", rough_correct_num / y_test.shape[0])

        stock_operation(stock_name=stock,
                        change=change_total[train_days:],
                        close=close_total[train_days:],
                        mean=(turnover_value_total * 10000 / turnover_vol_total)[train_days + 1:],
                        predict_state=y_predict)

        print("结束 :", stock, ", =====================================时间 :", time.ctime())
    except Exception as e:
        print(e)


def generate_x_and_y(time_steps, change, *xs):
    # ( ,-big_num)大跌， (-big_num,-small_num)小跌， (-small_num,small_num)平， (small_num,big_num)小涨， (big_num, )大涨
    small_num = 0.5
    big_num = 1.5

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

    x_unscaled = np.column_stack(x_list)  # 训练集
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # """
    # fit_transform()对部分数据先拟合fit，
    # 找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），
    # 然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等。
    # """
    # x_scaled = scaler.fit_transform(X=x_unscaled)
    x_scaled = scale(X=x_unscaled, axis=0)

    x = []
    y = []
    # 每240个数据为一组，作为测试数据，下一个数据为标签
    for i in range(time_steps, x_scaled.shape[0]):
        x.append(x_scaled[i - time_steps: i])
        y_change = change[i]
        if y_change < -big_num:
            y.append(0)
        elif -big_num <= y_change < -small_num:
            y.append(1)
        elif -small_num <= y_change <= small_num:
            y.append(2)
        elif small_num < y_change <= big_num:
            y.append(3)
        elif y_change > big_num:
            y.append(4)
    # 将数据转化为数组
    x, y = np.array(x), np.array(y)
    # 因为LSTM要求输入的数据格式为三维的，[training_number, time_steps, data_num]，因此对数据进行相应转化
    x = np.reshape(x, (x.shape[0], x.shape[1], data_num))
    y = np_utils.to_categorical(y, num_classes=5)

    return x, y


def stock_operation(stock_name, change, close, mean, predict_state):
    """
    股票买卖的操作，最后绘制走势图
    :param stock_name: 股票代码
    :param change: 涨跌幅
    :param close: 收盘价
    :param mean: 均价
    :param predict_state: 预测的每一天的状态
    :return:
    """
    base_money = close[0]  # 每天的金额，以第一天的前一天的收盘价为基础金额
    # close = close[1:]
    base_money_fee = base_money  # 每天的金额，含手续费计算
    base = 1  # 用来计算收益率
    base_fee = base  # 用来计算含手续费的收益率
    model_line = [base_money]  # 记录每天的金额的列表
    model_line_fee = [base_money]  # 记录含手续费的每天的金额的列表
    print(stock_name, '初始金额 : ', base_money, ' , 实际最终金额 : ', close[-1])

    buyed = 1  # 股票是否已经购买的状态
    buy_num = 0  # 购买股票的天数
    hold_num = 0  # 持有股票的天数
    sell_num = 0  # 抛出股票的天数
    empty_num = 0  # 空仓的天数

    up_num = 0  # 预测涨正确的天数
    down_num = 0  # 预测跌正确的天数
    medium_num = 0  # 预测平正确的天数

    for i in range(change.shape[0]):  # 从第一天起到最后一天，计算金额变化

        if predict_state[i] > 0 and change[i] > 2:
            up_num += 1
        elif predict_state[i] < 0 and change[i] < -2:
            down_num += 1
        elif predict_state[i] == 0 and -2 <= change[i] <= 2:
            medium_num += 1

        if predict_state[i] >= 0 and (not buyed):  # 预测结果不为跌 且 没有持有股票， 买入，手续费0.00032
            buyed = 1
            buy_num += 1
            rate_temp = (mean[i] - close[i]) / close[i]  # 基于第二天股票均价相对于第一天收盘价的涨跌幅
            base = base * (1 + rate_temp)
            base_money = base_money * (1 + rate_temp)
            base_fee = base_fee * (1 + rate_temp) * (1 - 0.00032)
            base_money_fee = base_money_fee * (1 + rate_temp) * (1 - 0.00032)
        elif predict_state[i] < 0 and buyed:  # 预测结果为跌 且 持有股票，抛出，手续费0.00132
            buyed = 0
            sell_num += 1
            rate_temp = (mean[i] - close[i]) / close[i]  # 基于第二天股票均价相对于第一天收盘价的涨跌幅
            base = base * (1 + rate_temp)
            base_money = base_money * (1 + rate_temp)
            base_fee = base_fee * (1 + rate_temp) * (1 - 0.00132)
            base_money_fee = base_money_fee * (1 + rate_temp) * (1 - 0.00132)
        elif predict_state[i] >= 0 and buyed:  # 预测结果为震荡或上涨 且 持有股票，不进行操作
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
          hold_num,',空仓天数: ', empty_num)
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
    plt.show()
    fig.savefig("pictures\\lines_" + stock_name + ".png")


if __name__ == '__main__':

    # secID,tradeDate,openPrice,closePrice,lowestPrice,highestPrice,accumAdjFactor(累积前复权因子),turnoverVol(成交量),turnoverValue(成交金额),negMarketValue(流通市值)
    quotes_dataframe = pd.read_hdf("datas/quotes.h5", "quotes")

    all_quotes = quotes_dataframe['secID'].unique()

    divide_date = '2019-01-01'  # 该时间之前的数据为训练集，之后的数据为测试集

    print("main 开始时间 :", time.ctime())

    # cpu_num = multiprocessing.cpu_count()
    # print("cpu num : ", cpu_num)

    pool = multiprocessing.Pool(processes=2)

    for stock in all_quotes[1:2]:
        quotes = quotes_dataframe[quotes_dataframe['secID'] == stock]
        pool.apply_async(stock_predict, (stock, quotes, divide_date))

    pool.close()
    pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束