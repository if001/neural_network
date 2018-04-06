

class AutoEncoderBase():
    def __init__(self, model, load_flag=""):
        self.model = model
        # tb_cb = TensorBoard(log_dir="~/tflog/", histogram_freq=1)
        # self.cbks = [tb_cb]
        if load_flag == "load":
            self.encoder = self.load_wight(self.model.encoder_weight)
            self.decoder = self.load_wight(self.model.decoder_weight)
        else:
            self.encoder = self.model.build_encoder()
            self.decoder = self.model.build_decoder()
        self.auto_encoder = self.model.build_autoencoder(self.encoder,
                                                         self.decoder)


    def model_train(self, train, teach):
        self.auto_encoder.compile(optimizer=self.model.optimizer,
                           loss=self.model.loss,
                           metrics=self.model.metrics)
        self.model.network.summary()
        loss = self.auto_encoder.fit(train, teach,
                              batch_size=self.model.batch_size,
                              epochs=self.model.epochs,
                              validation_split=self.model.validation_split)

    def model_predict(self): pass


    def load_wight(self, weight):
        from keras.models import load_model
        print("load "+weight)
        return load_model(weight)

    def save_wight(self):
        print("save"+self.model.weight)
        self.model.network.save(self.model.weight)
