import tensorflow as tf
from tensorflow import keras
from config.config import RNN_UNITS, MAX_LABEL_LEN
from tensorflow.keras import layers
from tensorflow.keras.backend import ctc_batch_cost

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        return y_pred

def build_crnn_ctc(img_height, img_width, num_classes, rnn_units=RNN_UNITS, max_label_len=MAX_LABEL_LEN):
    input_img = layers.Input(shape=(img_height, img_width, 1), name="image", dtype="float32")
    labels = layers.Input(name="label", shape=(max_label_len,), dtype="int32")

    # CNN backbone 
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(input_img)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)

    x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)

    x = layers.Conv2D(256, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3,3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,1))(x)

    x = layers.Conv2D(512, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(512, (3,3), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,1))(x)


    x = layers.Permute((2,1,3))(x)  # (batch, W, H, C)
    x = layers.Reshape((-1, x.shape[2]*x.shape[3]))(x)  #-> (batch, W, H*C)

    # RNN
    x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))(x)

    # Softmax output
    y_pred = layers.Dense(num_classes+1, activation="softmax", name="y_pred")(x)

    # CTC loss layer
    out = CTCLayer(name="ctc_loss")(labels, y_pred)

    train_model = keras.models.Model(inputs=[input_img, labels], outputs=out)
    pred_model = keras.models.Model(inputs=input_img, outputs=y_pred)

    return train_model, pred_model

