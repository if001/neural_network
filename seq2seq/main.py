from seq2seq import Seq2Seq
import sys
sys.path.append("../")
from lib.auto_encoder_base import AutoEncoderBase
from lib.auto_encoder_base import AutoEncoderBase

def main():
    # train data
    train_data, teach_data = 
    
    # train
    auto_encoder = AutoEncoderBase(Seq2Seq(128, 128))
    auto_encoder.model_train()

if __name__ == '__main__':
    main()
