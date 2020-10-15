import shift

print(shift.__version__)

import shift
from threading import Thread, Lock, Event
import pandas as pd
import time

trader = shift.Trader("test003")
trader.connect("initiator.cfg", "password")
trader.subAllOrderBook()

trader.disconnect()

order = shift.Order(shift.Order.LIMIT_BUY, 'AAPL',size = 10, price = 100.30)
trader.submitOrder(order)

order_ = trader.getOrder(order.id)
print(order_.status)

bp = trader.getBestPrice("AAPL")
bp.getBidPrice()
bp.getAskPrice()


trader.submitCancellation(order)


waitingList()

clearPosition()

portfolio()

executedOrder(order_.id)

def waitingList():
    print("Symbol\t\t\t\t\t Type\t  Price\t\tSize\tID\t\t\t\t\t\t\t\t\t\tTimestamp")
    for order in trader.getWaitingList():
        print("%6s\t%21s\t%7.2f\t\t%4d\t%36s\t%26s" %
              (order.symbol, order.type, order.price, order.size, order.id, order.timestamp))


def clearPosition():
    share = trader.getPortfolioItem('AAPL').getShares()
    waitingStep = 0
    while share != 0:
        print(share)
        position = int(share / 100)
        orderType = shift.Order.MARKET_BUY if position < 0 else shift.Order.MARKET_SELL
        order = shift.Order(orderType, 'AAPL', abs(position))
        trader.submitOrder(order)
        time.sleep(0.5)
        share = trader.getPortfolioItem('AAPL').getShares()
        waitingStep += 1
        assert waitingStep < 40

def portfolio():
    for item in trader.getPortfolioItems().values():
        print((item.getSymbol(), item.getShares(), item.getPrice()))


def executedOrder(id):
    print("Symbol\t\t\t\tType\t  Price\t\tSize\tExecuted\tID\t\t\t\t\t\t\t\t\t\t\t\t\t\t Status\t\tTimestamp")
    for order in trader.getExecutedOrders(id):
        print("%6s\t%16s\t%7.2f\t\t%4d\t\t%4d\t%36s\t%23s\t\t%26s" %
              (order.symbol, order.type, order.executed_price, order.size,
               order.executed_size, order.id, order.status, order.timestamp))
