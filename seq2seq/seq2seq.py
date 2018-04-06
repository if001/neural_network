import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, RepeatVector, Dense
from keras.optimizers import Adam,SGD,RMSprop
from keras.callbacks import EarlyStopping, TensorBoard
from keras import backend as K

class Seq2Seq():
    def __init__(self, input_dim, output_dim):
        self.optimizer = 'rmsprop'
        self.loss = 'mean_squared_error'
        self.batch_size = 10
        self.epochs = 10
        self.validation_split = 0.2
        self.metrics = ['accuracy']
        self.encoder_weight = ""
        self.decoder_weight = ""
        self.input_dim = input_dim
        self.output_dim = output_dim


    def build_encoder(self):
        latent_dim = 256

        encoder_inputs = Input(shape=(None, self.input_dim))
        encoder_dense_outputs = Dense(self.input_dim,
                                      activation='sigmoid')(encoder_inputs)
        encoder_lstm_outputs = LSTM(latent_dim, return_sequences=True,
                               dropout=0.6, recurrent_dropout=0.6)(encoder_dense_outputs)
        _, state_h, state_c = LSTM(latent_dim, return_state=True,
                                   dropout=0.2, recurrent_dropout=0.2)(encoder_lstm_outputs)
        return Model(encoder_inputs, [state_h, state_c])


    def build_decoder(self):
        K.set_learning_phase(1) # set learning phase
        latent_dim = 256
        encoder_h = Input(shape=(latent_dim,))
        encoder_c = Input(shape=(latent_dim,))
        encoder_states = [encoder_h, encoder_c]

        decoder_inputs = Input(shape=(None, self.input_dim))
        decoder_dense_outputs = Dense(self.input_dim, activation='sigmoid')(decoder_inputs)
        # decoder_lstm = LSTM(self.latent_dim, return_sequences=True,
        #                     dropout=0.6, recurrent_dropout=0.6)
        # decoder_lstm_outputs = decoder_lstm(decoder_dense_outputs)
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_lstm_outputs, _, _ = decoder_lstm(decoder_dense_outputs,
                                             initial_state=encoder_states)
        decoder_dense_outputs = Dense(self.output_dim,
                                      activation='relu')(decoder_lstm_outputs)
        decoder_outputs = Dense(self.output_dim,
                                activation='linear')(decoder_dense_outputs)

        return Model([decoder_inputs, encoder_h, encoder_c], decoder_outputs)


    def build_autoencoder(self, encoder, decoder):
        # encoder
        encoder_inputs = Input(shape=(None, self.input_dim))
        _, ed, el, el2 = encoder.layers
        dense_outputs = ed(encoder_inputs)
        el_outputs = el(dense_outputs)
        _, state_h, state_c = el2(el_outputs)
        encoder_states = [state_h, state_c]


        # decoder
        decoder_inputs = Input(shape=(None, self.input_dim))
        _, dd1, _, _, dl, dd2, dd3 = decoder.layers
        decoder_dense_outputs = dd1(decoder_inputs)
        decoder_lstm_outputs, _, _ = dl(decoder_dense_outputs,
                                        initial_state=encoder_states)
        decoder_outputs = dd2(decoder_lstm_outputs)
        outputs = dd3(decoder_outputs)

        return Model([encoder_inputs, decoder_inputs], outputs)


