from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization, Activation, Dropout


class MLModel:

    def __init__(self, input_shape, stock_name="test"):
        self.stock_name = stock_name  # 股票代码
        self.model = MLModel.build_model(input_shape)

    @staticmethod
    def build_model(input_shape):
        # build recurrent neural network
        model = Sequential()
        # return_sequences=True返回的是全部输出
        model.add(LSTM(units=64, input_shape=input_shape))
        # model.add(BatchNormalization())
        # model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
        # model.add(Dropout(0.2))
        model.add(Dense(units=3))
        model.add(Activation('softmax'))
        # add optimizer and loss function
        model.compile(optimizer='adam',loss='mean_squared_error', metrics=['accuracy'])
        # model.summary()
        return model

    def train_model(self, x_train, y_train, epoch=5, batch_size=None, verbose=1):  # verbose=0不显示训练过程
        self.model.fit(x=x_train, y=y_train, epochs=epoch, batch_size=batch_size, verbose=verbose)

    def predict(self, x_test):
        y_predict = self.model.predict(x_test, batch_size=None)
        return y_predict