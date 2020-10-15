import pandas as pd
import numpy as np
import shift
import time


def portfolioItems(trader):
    pi = trader.getPortfolioItems().values()
    idx = 0
    df_pi = pd.DataFrame(columns=['Symbol', 'Share', 'Close_Price', 'Trade_Price'])
    for order in pi:
        df_pi.loc[idx, ['Symbol', 'Share', 'Trade_Price']] = [order.getSymbol(), order.getShares(), abs(order.getPrice())]
        idx += 1
        df_pi = df_pi[df_pi['Share'] != 0]

        df_pi.index = np.arange(0, len(df_pi))
    for i in range(0, len(df_pi.index)):
        df_pi.loc[i, ['Close_Price']] = trader.getClosePrice(df_pi["Symbol"][i], df_pi["Share"][i] < 0,
                                                             int(abs(df_pi["Share"][i] / 100)))
    return df_pi



def portfolioValue(trader):
    portfolio = portfolioItems(trader)
    pnl = trader.getPortfolioSummary().getTotalRealizedPL()
    return round(1000000 + ((portfolio.Close_Price - portfolio.Trade_Price) * portfolio.Share).sum() + pnl, 2)





if __name__=='__main__':

    trader = shift.Trader("test003")
    #trader.disconnect()
    trader.connect("initiator.cfg", "password")
    trader.subAllOrderBook()

    trader.submitOrder(shift.Order(shift.Order.MARKET_BUY, "AAPL", 1, 0.00))
    trader.submitOrder(shift.Order(shift.Order.MARKET_SELL, "IBM", 1, 0.00))
    trader.submitOrder(shift.Order(shift.Order.LIMIT_BUY, "IBM", 1, 10.00))

    time.sleep(2)

    print("Symbol\t\tShares\t\tPrice\t\tP&L\t\tTimestamp")
    for item in trader.getPortfolioItems().values():
        print("%6s\t\t%6d\t%9.2f\t%7.2f\t\t%26s" %
              (item.getSymbol(), item.getShares(), item.getPrice(), item.getRealizedPL(), item.getTimestamp()))

    print("Symbol\t\t\t\t\t Type\t  Price\t\tSize\tID\t\t\t\t\t\t\t\t\t\tTimestamp")
    for order in trader.getWaitingList():
        print("%6s\t%21s\t%7.2f\t\t%4d\t%36s\t%26s" %
              (order.symbol, order.type, order.price, order.size, order.id, order.timestamp))

    print(trader.getPortfolioSummary().getTotalBP())


    print(portfolioItems(trader))
    print(portfolioValue(trader))
    trader.disconnect()
