# coding=utf-8
import numpy as np
import tushare as ts
import talib as ta
import time
import logging
import os
from sklearn.preprocessing import scale
from keras.utils import np_utils
import matplotlib.pyplot as plt
from ml_model import MLModel


def window_predict(stock, window_len, time_steps, flat_num, close, change, mean, data):
    """
    滑动窗口的频繁训练预测
    :param stock: 股票
    :param window_len: 窗口长度
    :param time_steps:  样本长度
    :param flat_num: 涨跌平的区分值
    :param close: 收盘价
    :param change: 涨跌幅
    :param mean: 均价
    :param data: 样本数据
    :return:
    """

    # 设置日志
    logs_dir = os.path.join(os.path.curdir, "logs")
    if os.path.exists(logs_dir) and os.path.isdir(logs_dir):
        pass
    else:
        os.mkdir(logs_dir)
    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = '%Y/%m/%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()  # 输出到控制台的handler
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')
    fhlr = logging.FileHandler(str('logs/' + stock + '.txt'), encoding='utf-8')  # 输出到文件的handler
    fhlr.setFormatter(formatter)
    fhlr.setLevel('INFO')
    logger.addHandler(chlr)
    logger.addHandler(fhlr)

    logging.info("stock:" + stock + ",window_len:" + str(window_len) + ",time_steps:" + str(time_steps)
                 + ",flat_num:" + str(flat_num))

    program_start_time = time.time()

    # 初始曲线的数据
    base_money = close[window_len - 1]  # 每天的金额，以第一天的前一天的收盘价为基础金额
    base_money_fee = base_money  # 每天的金额，含手续费计算
    base = 1  # 用来计算收益率
    base_fee = base  # 用来计算含手续费的收益率
    model_line = [base_money]  # 记录每天的金额的列表
    model_line_fee = [base_money]  # 记录含手续费的每天的金额的列表

    buyed = 1  # 股票是否已经购买的状态

    buy_num = 0  # 购买股票的天数
    hold_num = 0  # 持有股票的天数
    sell_num = 0  # 抛出股票的天数
    empty_num = 0  # 空仓的天数

    up_num = 0  # 预测涨正确的天数
    down_num = 0  # 预测跌正确的天数
    medium_num = 0  # 预测平正确的天数
    actual_up_num = 0  # 涨的天数
    actual_down_num = 0  # 跌的天数
    actual_medium_num = 0  # 平的天数

    predict_up_num_2 = 0
    predict_down_num_2 = 0
    actual_up_num_2 = 0
    actual_down_num_2 = 0

    predict_result = []
    actual_result = []

    for i in range(close.shape[0] - window_len):  # 窗口滑动，计算金额变化

        once_start_time = time.time()

        x_total, y_total = generate_x_and_y_3(flat_num, time_steps, change[i:i + window_len + 1],
                                              data[i:i + window_len + 1])

        x_train = x_total[0:window_len - time_steps]
        y_train = y_total[0:window_len - time_steps]
        x_test = x_total[window_len - time_steps:]
        y_test = y_total[window_len - time_steps:]
        y_test = np.argmax(y_test, axis=1) - 1
        actual_state = y_test[0]
        actual_result.append(actual_state)

        # 构建模型
        model = MLModel(input_shape=(x_train.shape[1], x_train.shape[2]), stock_name=stock)

        # 训练模型
        model.train_model(x_train=x_train, y_train=y_train, epoch=10, batch_size=4, verbose=0)  # verbose=0不显示训练过程

        # 预测结果
        y_predict = model.predict(x_test)
        y_predict = np.argmax(y_predict, axis=1) - 1
        predict_state = y_predict[0]

        predict_result.append(predict_state)

        if change[i + window_len] >= 0:
            actual_up_num_2 += 1
            if (predict_state > 0 and (not buyed)) or (predict_state >= 0 and buyed) :
                predict_up_num_2 += 1
        else:
            actual_down_num_2 += 1
            if (predict_state < 0 and buyed) or (predict_state <= 0 and not buyed):
                predict_down_num_2 += 1

        if actual_state > 0:
            actual_up_num += 1
            if predict_state > 0:
                up_num += 1
        elif actual_state < 0:
            actual_down_num += 1
            if predict_state < 0:
                down_num += 1
        elif actual_state == 0:
            actual_medium_num += 1
            if predict_state == 0:
                medium_num += 1

        if predict_state > 0 and (not buyed):  # 预测结果涨 且 没有持有股票， 买入，手续费0.00032
            buyed = 1
            buy_num += 1
            rate_temp = (mean[i + window_len] - close[i + window_len - 1]) / close[
                i + window_len - 1]  # 基于第二天股票均价相对于第一天收盘价的涨跌幅
            base = base * (1 + rate_temp)
            base_money = base_money * (1 + rate_temp)
            base_fee = base_fee * (1 + rate_temp) * (1 - 0.00032)
            base_money_fee = base_money_fee * (1 + rate_temp) * (1 - 0.00032)
        elif predict_state < 0 and buyed:  # 预测结果为跌 且 持有股票，抛出，手续费0.00132
            buyed = 0
            sell_num += 1
            rate_temp = (mean[i + window_len] - close[i + window_len - 1]) / close[
                i + window_len - 1]  # 基于第二天股票均价相对于第一天收盘价的涨跌幅
            base = base * (1 + rate_temp)
            base_money = base_money * (1 + rate_temp)
            base_fee = base_fee * (1 + rate_temp) * (1 - 0.00132)
            base_money_fee = base_money_fee * (1 + rate_temp) * (1 - 0.00132)
        elif predict_state >= 0 and buyed:  # 预测结果为涨 且 持有股票，不进行操作
            hold_num += 1
            base = base * (1 + change[i + window_len] / 100)
            base_money = base_money * (1 + change[i + window_len] / 100)
            base_fee = base_fee * (1 + change[i + window_len] / 100)
            base_money_fee = base_money_fee * (1 + change[i + window_len] / 100)
        else:  # 预测结果为跌 且 没有持有股票，不进行操作
            empty_num += 1

        model_line.append(base_money)
        model_line_fee.append(base_money_fee)

        once_end_time = time.time()
        logging.info('总次数:' + str(close.shape[0] - window_len) + ',当前第' + str(i + 1) + '次循环用时:'
                     + str(once_end_time - once_start_time) + '秒')

    model_line = np.array(model_line)
    model_line_fee = np.array(model_line_fee)

    # result_compare = np.column_stack([predict_result, actual_result]).transpose()
    # print('预测结果序列, 实际结果序列')
    # print(result_compare)
    predict_result = np.array(predict_result)
    actual_result = np.array(actual_result)
    logging.info('预测结果序列' + str(predict_result))
    logging.info('实际结果序列' + str(actual_result))

    logging.info(stock + '预测涨正确的比例' + str(up_num / actual_up_num * 100) + '%')
    logging.info(stock + '预测跌正确的比例' + str(down_num / actual_down_num * 100) + '%')
    logging.info(stock + '预测平正确的比例' + str(medium_num / actual_medium_num * 100) + '%')

    logging.info(stock + '操作涨正确的比例' + str(predict_up_num_2 / actual_up_num_2 * 100) + '%')
    logging.info(stock + '操作跌正确的比例' + str(predict_down_num_2 / actual_down_num_2 * 100) + '%')

    logging.info(stock + '总天数:' + str(close.shape[0] - window_len) + ',买入天数: ' + str(buy_num) + ',卖出天数: ' + str(
        sell_num) + ',持有天数: ' + str(hold_num) + ',空仓天数: ' + str(empty_num))
    logging.info(stock + '初始金额 : ' + str(close[window_len - 1]) + ' , 实际最终金额 : ' + str(close[-1]))
    logging.info(stock + '最终金额: ' + str(base_money) + ' , 最终金额(含交易费): ' + str(base_money_fee))
    logging.info(stock + "收益率: " + str(base - 1) + " , 收益率(含交易费): " + str(base_fee - 1))

    program_end_time = time.time()
    logging.info('程序总用时:' + str((program_end_time - program_start_time) / 60) + '分钟')

    # 绘制曲线图
    fig = plt.figure()
    plt.plot(close[window_len - 1:], color='green', label='Real Stock Price')
    plt.plot(model_line, color='red', label='Predicted Without Fee')
    plt.plot(model_line_fee, color='yellow', label='Predicted With Fee')
    plt.title(label='Stock Prediction')
    plt.xlabel(xlabel='Time')
    plt.ylabel(ylabel='Stock Price')
    plt.legend(loc='upper left')
    plt.show()
    fig.savefig("pictures\\window_" + stock + ".png")


def generate_x_and_y_3(flat_num, time_steps, change, data):
    x = []
    y = []
    for i in range(time_steps, data.shape[0]):
        x.append(data[i - time_steps: i])
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
    x = np.reshape(x, (x.shape[0], x.shape[1], data.shape[1]))
    y = np_utils.to_categorical(y, num_classes=3)

    return x, y


def window_back(stock, window_len, time_steps, flat_num, close, change, mean, predict_state, actual_state):
    # 设置日志
    logs_dir = os.path.join(os.path.curdir, "logs")
    if os.path.exists(logs_dir) and os.path.isdir(logs_dir):
        pass
    else:
        os.mkdir(logs_dir)
    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = '%Y/%m/%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()  # 输出到控制台的handler
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')
    fhlr = logging.FileHandler(str('logs/' + stock + '.txt'), encoding='utf-8')  # 输出到文件的handler
    fhlr.setFormatter(formatter)
    fhlr.setLevel('INFO')
    logger.addHandler(chlr)
    logger.addHandler(fhlr)

    logging.info("stock:" + stock + ",window_len:" + str(window_len) + ",time_steps:" + str(time_steps)
                 + ",flat_num:" + str(flat_num))

    # 初始曲线的数据
    base_money = close[window_len - 1]  # 每天的金额，以第一天的前一天的收盘价为基础金额
    base_money_fee = base_money  # 每天的金额，含手续费计算
    base = 1  # 用来计算收益率
    base_fee = base  # 用来计算含手续费的收益率
    model_line = [base_money]  # 记录每天的金额的列表
    model_line_fee = [base_money]  # 记录含手续费的每天的金额的列表

    buyed = 1  # 股票是否已经购买的状态

    buy_num = 0  # 购买股票的天数
    hold_num = 0  # 持有股票的天数
    sell_num = 0  # 抛出股票的天数
    empty_num = 0  # 空仓的天数

    up_num = 0  # 预测涨正确的天数
    down_num = 0  # 预测跌正确的天数
    medium_num = 0  # 预测平正确的天数
    actual_up_num = 0  # 涨的天数
    actual_down_num = 0  # 跌的天数
    actual_medium_num = 0  # 平的天数

    for i in range(close.shape[0] - window_len):  # 窗口滑动，计算金额变化

        if actual_state[i] > 0:
            actual_up_num += 1
            if predict_state[i] > 0:
                up_num += 1
        elif actual_state[i] < 0:
            actual_down_num += 1
            if predict_state[i] < 0:
                down_num += 1
        elif actual_state[i] == 0:
            actual_medium_num += 1
            if predict_state[i] == 0:
                medium_num += 1

        if predict_state[i] > 0 and (not buyed):
            # 预测结果涨 且 没有持有股票， 买入，手续费0.00032
            buyed = 1
            buy_num += 1
            rate_temp = (mean[i + window_len] - close[i + window_len - 1]) / close[
                i + window_len - 1]  # 基于第二天股票均价相对于第一天收盘价的涨跌幅
            base = base * (1 + rate_temp)
            base_money = base_money * (1 + rate_temp)
            base_fee = base_fee * (1 + rate_temp) * (1 - 0.00032)
            base_money_fee = base_money_fee * (1 + rate_temp) * (1 - 0.00032)
        elif predict_state[i] < 0 and buyed:
            # 预测结果为跌 且 持有股票，抛出，手续费0.00132
            buyed = 0
            sell_num += 1
            rate_temp = (mean[i + window_len] - close[i + window_len - 1]) / close[
                i + window_len - 1]  # 基于第二天股票均价相对于第一天收盘价的涨跌幅
            base = base * (1 + rate_temp)
            base_money = base_money * (1 + rate_temp)
            base_fee = base_fee * (1 + rate_temp) * (1 - 0.00132)
            base_money_fee = base_money_fee * (1 + rate_temp) * (1 - 0.00132)
        elif predict_state[i] >= 0 and buyed:
            # 预测结果为涨 且 持有股票，不进行操作
            hold_num += 1
            base = base * (1 + change[i + window_len] / 100)
            base_money = base_money * (1 + change[i + window_len] / 100)
            base_fee = base_fee * (1 + change[i + window_len] / 100)
            base_money_fee = base_money_fee * (1 + change[i + window_len] / 100)
        else:  # 预测结果为跌 且 没有持有股票，不进行操作
            empty_num += 1

        model_line.append(base_money)
        model_line_fee.append(base_money_fee)

    model_line = np.array(model_line)
    model_line_fee = np.array(model_line_fee)

    # result_compare = np.column_stack([predict_result, actual_result]).transpose()
    # print('预测结果序列, 实际结果序列')
    # print(result_compare)
    predict_result = np.array(predict_state)
    actual_result = np.array(actual_state)
    logging.info('预测结果序列' + str(predict_result))
    logging.info('实际结果序列' + str(actual_result))

    logging.info(stock + '预测涨正确的比例' + str(up_num / actual_up_num * 100) + '%')
    logging.info(stock + '预测跌正确的比例' + str(down_num / actual_down_num * 100) + '%')
    logging.info(stock + '预测平正确的比例' + str(medium_num / actual_medium_num * 100) + '%')

    logging.info(stock + '总天数:' + str(close.shape[0] - window_len) + ',买入天数: ' + str(buy_num) + ',卖出天数: ' + str(
        sell_num) + ',持有天数: ' + str(hold_num) + ',空仓天数: ' + str(empty_num))
    logging.info(stock + '初始金额 : ' + str(close[window_len - 1]) + ' , 实际最终金额 : ' + str(close[-1]))
    logging.info(stock + '最终金额: ' + str(base_money) + ' , 最终金额(含交易费): ' + str(base_money_fee))
    logging.info(stock + "收益率: " + str(base - 1) + " , 收益率(含交易费): " + str(base_fee - 1))

    # 绘制曲线图
    fig = plt.figure()
    plt.plot(close[window_len - 1:], color='green', label='Real Stock Price')
    plt.plot(model_line, color='red', label='Predicted Without Fee')
    plt.plot(model_line_fee, color='yellow', label='Predicted With Fee')
    plt.title(label='Stock Prediction')
    plt.xlabel(xlabel='Time')
    plt.ylabel(ylabel='Stock Price')
    plt.legend(loc='upper left')
    plt.show()
    fig.savefig("pictures\\window_" + stock + ".png")


if __name__ == '__main__':
    f = open("token.txt", "r")
    token = f.read()
    f.close()
    ts.set_token(token)
    pro = ts.pro_api()

    # 参数
    stock = '399300.SZ'  # 399300.SZ   000905.SH
    window_len = 20  # 窗口的长度
    time_steps = 3  # 每次训练的数据长度
    flat_num = 0.4  # (-flat_num, flat_num) 平

    quotes = pro.index_daily(ts_code=stock, start_date='20190301').sort_values(by=['trade_date'], ascending=True)

    close_total = np.array(quotes['close'])
    open_total = np.array(quotes['open'])
    high_total = np.array(quotes['high'])
    low_total = np.array(quotes['low'])
    change_total = np.array(quotes['pct_chg'])
    turnover_vol_total = np.array(quotes['vol'])
    turnover_value_total = np.array(quotes['amount'])
    mean_total = (np.array(quotes['high']) + np.array(quotes['low'])) / 2  # 均价

    # 数据ema5 ema10 穿越 macd rsi (diff   kdj  布林线
    # 金融数据指标
    ema_num = 5
    boll_num = 20
    ema = ta.EMA(quotes['close'], ema_num).values
    H_line, M_line, L_line = ta.BBANDS(quotes['close'], timeperiod=boll_num, nbdevup=2, nbdevdn=2, matype=0)

    data_list = [ema[boll_num - 1:], H_line[boll_num - 1:], M_line[boll_num - 1:], L_line[boll_num - 1:],
                 close_total[boll_num - 1:], turnover_vol_total[boll_num - 1:]]
    data_unscaled = np.column_stack(data_list)
    data_scaled = scale(X=data_unscaled, axis=0)  # 归一化

    # 回测，减少时间
    # predict_result = [1, 1, 1, -1, 0, 0, -1, 0, 1, 0, -1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1,
    #                   -1, -1, -1, 1, 1, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, -1, 0, 0, 0, 0,
    #                   0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1]
    # actual_result = [1, 0, -1, 0, 0, -1, 0, -1, 0, 0, 1, 0, 0, -1, 1, -1, 1, 1, 0, 0, 0, 1, -1, -1, -1, -1, -1, -1, 1,
    #                  -1, 1, -1, 1, 1, 0, 1, 0, 0, 0, 0, -1, 1, 0, 0, -1, 1, 1, 1, 1, 0, 1, 0, -1, 1, 0, -1, 0, 1, 0, -1,
    #                  0, -1, -1, 1, -1, 0, 1, 1, 0, 1, -1, -1, 0, -1, 0, 1, -1, 0, 1, 1]
    # window_back(stock=stock, window_len=window_len, time_steps=time_steps, flat_num=flat_num,
    #             close=close_total[boll_num - 1:], change=change_total[boll_num - 1:], mean=mean_total[boll_num - 1:],
    #             predict_state=predict_result, actual_state=actual_result)
    # exit()

    window_predict(stock=stock, window_len=window_len, time_steps=time_steps, flat_num=flat_num,
                   close=close_total[boll_num - 1:], change=change_total[boll_num - 1:], mean=mean_total[boll_num - 1:],
                   data=data_scaled)
