import pandas as pd
import numpy as np
from features import Features
from misc import Recorder
from market import Market

def main():
    mkt = Market(date = '20190716', 
                 instrument = 'sc',
                 trade_fee_ratio = 0.000025,
                 threshold = 0.75,
                 reverse_position = 8,
                 MAX_BID = 2)
    


if __name__=='__main__':
    pass
    