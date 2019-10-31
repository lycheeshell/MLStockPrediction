from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout, Convolution1D, MaxPooling1D, Flatten
from keras.optimizers import Adam


class MLModel:

    def __init__(self, input_shape, stock_name="test"):
        self.stock_name = stock_name  # 股票代码
        self.model = MLModel.build_model(input_shape)

    @staticmethod
    def build_model(input_shape):
        # build recurrent neural network
        model = Sequential()
        # return_sequences=True返回的是全部输出
        model.add(Convolution1D(
            input_shape=input_shape,
            filters=32,
            kernel_size=5,
            strides=1,
            padding='same',  # Padding method
            data_format='channels_first',
        ))
        model.add(MaxPooling1D(
            pool_size=2,
            strides=2,
            padding='same',  # Padding method
            data_format='channels_first',
        ))
        model.add(LSTM(units=64))
        # model.add(BatchNormalization())
        # model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
        # model.add(Dropout(0.2))
        model.add(Dense(units=3))
        model.add(Activation('softmax'))
        # add optimizer and loss function
        adam = Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        # model.summary()
        return model

    def train_model(self, x_train, y_train, epoch=5, batch_size=None, verbose=1):  # verbose=0不显示训练过程
        self.model.fit(x=x_train, y=y_train, epochs=epoch, batch_size=batch_size, verbose=verbose)

    def predict(self, x_test):
        y_predict = self.model.predict(x_test, batch_size=None)
        return y_predict
