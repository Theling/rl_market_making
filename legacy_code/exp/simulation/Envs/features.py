from .misc import feature, FeaturesBase
import pandas as pd


def get_time_roll_return(pxs, look_back_secs=1):
    """ get the time based return
    Args:
        pxs:    price series
        look_back_secs: look back period in seconcds
    Return:
        a series of roll return
    """
    look_back_ns = look_back_secs * 1000000000
    px_diff = pd.Series( \
        (pxs.values - pxs.asof(pxs.index-look_back_ns).values) / pxs.asof(pxs.index-look_back_ns).values, \
        pxs.index)

    return px_diff

class Features(FeaturesBase):
    def __init__(self):
        FeaturesBase.__init__(self)

    @feature
    def BidAskSpread(self, data):
        spread = data.ap0 - data.bp0
        return spread

    @feature
    def vwap_mp(self, data):
        mp = (data.ap0 + data.bp0) / 2
        vwap = data.vwap.fillna(method='bfill')
        return vwap - mp

    @feature
    def volume4(self, data):
        v = data.v_diff.rolling(4).sum()
        return v

    def volume10(self, data):
        v = data.v_diff.rolling(10).sum()
        return v

    @feature
    def market_pressure1(self, data):
        ask_pressure = data.az0 + data.az1 + data.az2 + data.az3 + data.az4
        bid_pressure = data.bz0 + data.bz1 + data.bz2 + data.bz3 + data.bz4
        market_pressure_all_ratio = (ask_pressure - bid_pressure) / (ask_pressure + bid_pressure)
        return market_pressure_all_ratio

    @feature
    def market_pressure2(self, data, alpha=2):
        market_pressure1 = alpha * data.bz0 + data.bz1 - alpha * data.az0 - data.az1
        sum_of_volume = alpha * data.bz0 + data.bz1 + alpha * data.az0 + data.az1
        market_pressure_ratio = market_pressure1 / sum_of_volume
        return market_pressure_ratio

    @feature
    def market_pressure0(self, data):
        market_pressure = data.bz0 - data.az0
        sum_of_volume = data.bz0 + data.az0
        market_pressure_ratio = market_pressure / sum_of_volume
        return market_pressure_ratio

    @feature
    def wp1_mp(self, data):
        mp = (data.ap0 + data.bp0) / 2
        wp1 = (data.ap0 * data.az0 + data.bp0 * data.bz0) / (data.az0 + data.bz0)
        return wp1 - mp

    @feature
    def wp2_mp(self, data):
        mp = (data.ap0 + data.bp0) / 2
        wp2 = (data.ap0 * data.bz0 + data.bp0 * data.az0) / (data.az0 + data.bz0)
        return wp2 - mp

    @feature
    def re_feature(self, data):
        mp = (data.ap0 + data.bp0) / 2
        return_val = get_time_roll_return(mp, 2 - 0.02)
        return return_val

    @feature
    def re_feature4(self, data):
        mp = (data.ap0 + data.bp0) / 2
        return_val = get_time_roll_return(mp, 2 - 0.02)
        return_val = return_val.rolling(4).sum()
        return return_val

    @feature
    def TFI(self, data):
        trade_imbalance = data.ask_take_order - data.bid_take_order
        return trade_imbalance

    @feature
    def Order_imbalance(self, data):
        condition_bid0 = (data.bp0 > data.bp0.shift(1))
        condition_bid1 = (data.bp0 == data.bp0.shift(1))
        condition_ask0 = (data.ap0 < data.ap0.shift(1))
        condition_ask1 = (data.ap0 == data.ap0.shift(1))
        data['bid_volume'] = 0
        data['ask_volume'] = 0
        data.loc[condition_bid1, 'bid_volume'] = data.bz0 - data.bz0.shift(1)
        data.loc[condition_bid0, 'bid_volume'] = data.bz0
        data.loc[condition_ask0, 'ask_volume'] = data.az0
        data.loc[condition_ask1, 'ask_volume'] = data.az0 - data.az0.shift(1)
        order_imbalance = data.bid_volume - data.ask_volume
        return order_imbalance

    @feature
    def macd_10(self, data):
        long = 10
        short = 10 // 2
        mp = (data.ap0 + data.bp0) / 2
        fas = mp.ewm(span=short).mean()
        slo = mp.ewm(span=long).mean()
        MACD = fas - slo
        return MACD

    @feature
    def macd_6(self, data):
        long = 6
        short = 6 // 2
        mp = (data.ap0 + data.bp0) / 2
        fas = mp.ewm(span=short).mean()
        slo = mp.ewm(span=long).mean()
        MACD = fas - slo
        return MACD

    @feature
    def bb_10(self, data):
        ndev = 2
        mp = (data.ap0 + data.bp0) / 2
        SMA = mp.rolling(10).mean()
        STD = mp.rolling(10).std()
        return (SMA - mp) / (2 * STD * ndev)

    @feature
    def bb_5(self, data):
        ndev = 2
        mp = (data.ap0 + data.bp0) / 2
        SMA = mp.rolling(5).mean()
        STD = mp.rolling(5).std()
        return (SMA - mp) / (2 * STD * ndev)

    def compute(self, data):
        return [func(self, data) for func in self.feature_list]


if __name__ == "__main__":
    from market import Market

    import os
    os.chdir('dev/Env')
    print(os.getcwd())

    mkt = Market('mkt_data/20190701',
                 'sc',
                 0.00025,
                 threshold=0,
                 reverse_position=0,
                 MAX_BID=0)

    feature_comp = Features() 
    print(feature_comp.feature_list)   
    for _ in range(10): mkt.next() # skip 10 ticks
    print(mkt.get_data(5)) # this is an one row data frame (5 rows). From time step t-5 to t (latest time step).
    data = mkt.get_data(5)
    '''
    Please make sure all the feature methods take the same data frame and 
    return a single value. Then this feature_comp.compute() will
    return a list of features for a single time step (for time 09:00:05.500 in this particular case).
    '''
    print(feature_comp.compute(data))
    # for _ in range(5):
    #     print()
    #     print(mkt.exch_time)
    #     data = mkt.get_data()
    #     print(feature_comp.compute(data))

    #     mkt.next()
