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


def classify_predict(stock, time_steps, close, change, mean, data, state, back_len):
    """
    滑动窗口的频繁训练预测
    :param stock: 股票
    :param time_steps:  样本长度
    :param close: 收盘价
    :param change: 涨跌幅
    :param mean: 均价
    :param data: 数据
    :param state: 涨跌平的状态序列
    :param back_len: 回测数据的长度
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

    logging.info("stock:" + stock + ",time_steps:" + str(time_steps))

    program_start_time = time.time()

    total_len = len(change)

    # 初始曲线的数据
    base_money = close[total_len - back_len - 1]  # 每天的金额，以第一天的前一天的收盘价为基础金额
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

    predict_result = []
    actual_result = []

    existed_state_list = []
    existed_model = []

    for i in range(back_len):  # 窗口滑动，计算金额变化

        once_start_time = time.time()

        state_list = state[total_len - back_len - time_steps + i: total_len - back_len + i]

        x_test = data[total_len - back_len - time_steps + i: total_len - back_len + i]
        x_test = x_test.reshape(-1,x_test.shape[0], x_test.shape[1])
        y_actual = state[total_len - back_len + i]
        actual_state = y_actual - 1
        actual_result.append(actual_state)

        model = None

        for k in range(len(existed_state_list)):
            li = existed_state_list[k]
            all_same = 1
            for j in range(time_steps):
                if li[j] != state_list[j]:
                    all_same = 0
                    break
            if all_same:
                model = existed_model[k]
                break

        if model is None:
            x_train, y_train = generate_x_and_y_3(data=data[0:total_len-back_len], state_list=state_list, state=state[0:total_len-back_len])

            # 构建模型
            model = MLModel(input_shape=(x_train.shape[1], x_train.shape[2]), stock_name=stock)

            # 训练模型
            model.train_model(x_train=x_train, y_train=y_train, epoch=10, batch_size=4, verbose=0)  # verbose=0不显示训练过程

            existed_model.append(model)
            existed_state_list.append(state_list)

        # 预测结果
        y_predict = model.predict(x_test)
        y_predict = np.argmax(y_predict, axis=1) - 1
        predict_state = y_predict[0]

        predict_result.append(predict_state)

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
            rate_temp = (mean[total_len - back_len + i] - close[total_len - back_len + i - 1]) / close[
                total_len - back_len + i - 1]  # 基于第二天股票均价相对于第一天收盘价的涨跌幅
            base = base * (1 + rate_temp)
            base_money = base_money * (1 + rate_temp)
            base_fee = base_fee * (1 + rate_temp) * (1 - 0.00032)
            base_money_fee = base_money_fee * (1 + rate_temp) * (1 - 0.00032)
        elif predict_state < 0 and buyed:  # 预测结果为跌 且 持有股票，抛出，手续费0.00132
            buyed = 0
            sell_num += 1
            rate_temp = (mean[total_len - back_len + i] - close[total_len - back_len + i - 1]) / close[
                total_len - back_len + i - 1]  # 基于第二天股票均价相对于第一天收盘价的涨跌幅
            base = base * (1 + rate_temp)
            base_money = base_money * (1 + rate_temp)
            base_fee = base_fee * (1 + rate_temp) * (1 - 0.00132)
            base_money_fee = base_money_fee * (1 + rate_temp) * (1 - 0.00132)
        elif predict_state >= 0 and buyed:  # 预测结果为涨 且 持有股票，不进行操作
            hold_num += 1
            base = base * (1 + change[total_len - back_len + i] / 100)
            base_money = base_money * (1 + change[total_len - back_len + i] / 100)
            base_fee = base_fee * (1 + change[total_len - back_len + i] / 100)
            base_money_fee = base_money_fee * (1 + change[total_len - back_len + i] / 100)
        else:  # 预测结果为跌 且 没有持有股票，不进行操作
            empty_num += 1

        model_line.append(base_money)
        model_line_fee.append(base_money_fee)

        once_end_time = time.time()
        logging.info('总次数:' + str(back_len) + ',当前第' + str(i + 1) + '次循环用时:'
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

    logging.info(stock + '预测涨准确率' + str(up_num / actual_up_num * 100) + '%')
    logging.info(stock + '预测跌准确率' + str(down_num / actual_down_num * 100) + '%')
    logging.info(stock + '预测平准确率' + str(medium_num / actual_medium_num * 100) + '%')

    logging.info(stock + '总天数:' + str(back_len) + ',买入天数: ' + str(buy_num) + ',卖出天数: ' + str(
        sell_num) + ',持有天数: ' + str(hold_num) + ',空仓天数: ' + str(empty_num))
    logging.info(stock + '初始金额 : ' + str(close[total_len - back_len - 1]) + ' , 实际最终金额 : ' + str(close[-1]))
    logging.info(stock + '最终金额: ' + str(base_money) + ' , 最终金额(含交易费): ' + str(base_money_fee))
    logging.info(stock + "收益率: " + str(base - 1) + " , 收益率(含交易费): " + str(base_fee - 1))

    program_end_time = time.time()
    logging.info('程序总用时:' + str((program_end_time - program_start_time) / 60) + '分钟')

    # 绘制曲线图
    fig = plt.figure()
    plt.plot(close[total_len - back_len - 1:], color='green', label='Real Stock Price')
    plt.plot(model_line, color='red', label='Predicted Without Fee')
    plt.plot(model_line_fee, color='yellow', label='Predicted With Fee')
    plt.title(label='Stock Prediction')
    plt.xlabel(xlabel='Time')
    plt.ylabel(ylabel='Stock Price')
    plt.legend(loc='upper left')
    plt.show()
    fig.savefig("pictures\\classify_" + stock + ".png")


def generate_state(change, flat_num):
    state = []
    for i in range(len(change)):
        if change[i] < -flat_num:
            state.append(0)
        elif -flat_num <= change[i] <= flat_num:
            state.append(1)
        elif change[i] > flat_num:
            state.append(2)
    state = np.array(state)

    return state


def generate_x_and_y_3(data, state_list, state):
    x = []
    y = []
    sample_len = len(state_list)
    for i in range(len(state) - sample_len -1):
        same = 1
        for k in range(sample_len):
            if state_list[k] != state[i + k]:
                same = 0
                break
        if same:
            x.append(data[i:i+sample_len])
            y.append(state[i+sample_len])
    # 将数据转化为数组
    x, y = np.array(x), np.array(y)
    # 因为LSTM要求输入的数据格式为三维的，[training_number, time_steps, data_num]，因此对数据进行相应转化
    # x = np.reshape(x, (x.shape[0], x.shape[1], data.shape[1]))
    y = np_utils.to_categorical(y, num_classes=3)

    return x, y


if __name__ == '__main__':
    f = open("token.txt", "r")
    token = f.read()
    f.close()
    ts.set_token(token)
    pro = ts.pro_api()

    # 参数
    stock = '000905.SH'  # 399300.SZ   000905.SH
    time_steps = 3  # 每次训练的数据长度
    flat_num = 0.4  # (-flat_num, flat_num) 平
    train_date = '20050101'
    back_date = '20180601'

    quotes = pro.index_daily(ts_code=stock, start_date=train_date).sort_values(by=['trade_date'], ascending=True)
    back_quotes = quotes[quotes['trade_date'] >= back_date].sort_values(by=['trade_date'], ascending=True)

    back_len = len(back_quotes['close'])

    close_total = np.array(quotes['close'])
    open_total = np.array(quotes['open'])
    high_total = np.array(quotes['high'])
    low_total = np.array(quotes['low'])
    change_total = np.array(quotes['pct_chg'])
    turnover_vol_total = np.array(quotes['vol']) / 10000
    turnover_value_total = np.array(quotes['amount']) / 10000
    mean_total = (np.array(quotes['high']) + np.array(quotes['low'])) / 2  # 均价

    state_total = generate_state(change_total, flat_num)

    # 数据ema5 ema10 穿越 macd rsi (diff   kdj  布林线
    # 金融数据指标
    ema_num = 5
    boll_num = 20
    ema = ta.EMA(quotes['close'], ema_num).values
    H_line, M_line, L_line = ta.BBANDS(quotes['close'], timeperiod=boll_num, nbdevup=2, nbdevdn=2, matype=0)

    data_list = [H_line - M_line, M_line - L_line, close_total - M_line, close_total, turnover_vol_total]
    data_unscaled = np.column_stack(data_list)[boll_num - 1:]
    data_scaled = scale(X=data_unscaled, axis=0)  # 归一化

    classify_predict(stock=stock, time_steps=time_steps,
                   close=close_total[boll_num - 1:], change=change_total[boll_num - 1:], mean=mean_total[boll_num - 1:],
                   data=data_scaled, state=state_total[boll_num - 1:], back_len=back_len)
