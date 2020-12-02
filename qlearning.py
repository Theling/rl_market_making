import sys
import time
import shift
import random
import numpy as np
import pandas as pd
import pickle
import pprint
import math
import shift

num_episodes = 10000

class Agent():
    def __init__(self,
                 a = 1,
                 alpha = 1,
                 b = 1,
                 epsilon = 0.2,
                 epsilon_decay = 0.995,
                 gamma = 0.1,
                 learn_rate = 0.01):

        self.a = a
        self.action_size = randint()
        self.action_size = 9
        self.alpha = alpha
        self.b = b
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.inventory = []
        self.learn_rate = learn_rate
        self.Q = np.array(np.zeros([self.state_size, self.action_size, self.timesteps]))
        self.state_size = 7
        self.states = []
        self.tick = 0.01
        self.time_count = 0
        self.time_current = 0
        self.timesteps = 12
        self.value = []

    def action(bid_tick, ask_tick):
        past_inventory = shift.PortfolioItem.get_shares()
        ask = shift.get_ask_price + ask_tick * tick
        bid = shift.get_bid_price + bid_tick * tick
        ask_order = shift.Order(shift.Order.Type.LIMIT_SELL)
        bid_order = shift.Order(shift.Order.Type.LIMIT_BUY)

    def action_choose(state):
        if random.uniform(0, 1) < epsilon:
            bid_tick = random.randint(0, 2)
            ask_tick = random.randint(0, 2)
            agent.action(bid_tick, ask_tick)
        else:
            optimal_action = np.argmax(Q[state, :, timestep])
        return bid_tick, ask_tick

    def cancel():
        trader.cancel_all_pending_orders()

    def get_action(bid_tick, ask_tick):
        if bid_tick = 0 && ask_tick = 0:
            action = 0
        elif bid_tick = 0 && ask_tick = 1:
            action = 1
        elif bid_tick = 0 && ask_tick = 2:
            action = 2
        elif bid_tick = 1 && ask_tick = 0:
            action = 3
        elif bid_tick = 1 && ask_tick = 1:
            action = 4
        elif bid_tick = 1 && ask_tick = 2:
            action = 5
        elif bid_tick = 2 && ask_tick = 0:
            action = 6
        elif bid_tick = 2 && ask_tick = 1:
            action = 7
        elif bid_tick = 2 && ask_tick = 2:
            action = 8
        return action

    def get_state():
        shares = shift.PortfolioItem.get_shares()
        if inventory < -400:
            state = 0
        elif inventory >= -400 && inventory < -200:
            state = 1
        elif inventory >= -200 && inventory < 0:
            state = 2
        elif inventory == 0:
            state = 3
        elif inventory > 0 && inventory <= 200:
            state = 4
        elif inventory > 200 && inventory <= 400:
            state = 5
        else:
            state = 6
        return state

    def get_time_remaining():
        return 10 * (self.timesteps - self.time_count)

    def inventory_update():
        inventory.append(shift.Portfolio.get_shares())
        return inventory[len(inventory) - 1], inventory[len(inventory) - 2]

    def Q_update(reward, state, new_state):
        action = get_action()
        timestep = self.time_count
        Q[state, action, timestep] = Q[state, action, timestep] + lr * (reward + gamma * np.max(Q[new_state, :, timestep]) - Q[state, action, timestep])
        # time stop for ten seconds

    def Q_update_terminal(reward, state):
        action = get_action()
        Q[state, action, timestep] = Q[state, action, timestep] + lr * reward

    def reward_immediate():
        current_inventory, past_inventory = inventory_update()
        current_value, past_value = value_update()
        tau = get_time_remaining()
        reward = a * (current_value - past_value) + math.exp(b * tau) * sgn(math.abs(current_inventory) - math.abs(past_inventory))
        return reward

    def reward_terminal():
        profit = trader.get_portfolio_summary().get_total_realize_pl()
        liquidate_value = shift.PortfolioItem.get_price() * shift.PortfolioItem.get_shares()
        reward = alpha - math.exp(-1 * r * (profit - liquidate_value))
        return reward

    def train(num_episodes):
        for episode in range(num_episodes+1):
            self.inventory = np.zeros(timesteps)
            self.value = np.zeros(timesteps)
            for timesteps in range(12):
                current_state = self.get_state()
                cancel()
                action_choose()
                time = get_time_remaining()
                new_state = self.get_state()
                reward = reward_immediate()
                Q_update(reward, current_state, new_state)
            reward_end = reward_terminal()
            Q_update_terminal(reward_end, new_state)

    def sgn(x):
        if x >= 0:
            return 1
        else:
            return -1

    def value_update():
        value.append(shift.PortfolioItem.get_shares() * shift.PortfolioItem.get_price())
        return value[len(value) - 1], value[len(value) - 2]

    def state_update(state):
        self.states.append(state)
        return self.states[len(self.states) - 1]

trader = shift.Trader("test003")
trader.connect("initial.cfg", "password")
trader.subAllOrderBook()
