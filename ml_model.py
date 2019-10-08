from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Activation


class MLModel:

    def __init__(self, input_shape, stock_name="test"):
        self.stock_name = stock_name  # 股票代码
        self.model = MLModel.build_model(input_shape)

    @staticmethod
    def build_model(input_shape):
        # build recurrent neural network
        model = Sequential()
        # return_sequences=True返回的是全部输出
        model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(LSTM(units=64))
        model.add(BatchNormalization())
        model.add(Dense(units=1))
        model.add(Activation('sigmoid'))
        # add optimizer and loss function
        model.compile(optimizer='adam',loss='mean_squared_error')
        return model

    def train_model(self, x_train, y_train, epoch=1, batch_size=None):
        self.model.fit(x=x_train, y=y_train, epochs=epoch, batch_size=batch_size)

    def train_once(self, x_train, y_train):
        self.model.fit(x=x_train, y=y_train)

    def predict(self, x_test):
        y_predict = self.model.predict(x_test, batch_size=32)
        return y_predict