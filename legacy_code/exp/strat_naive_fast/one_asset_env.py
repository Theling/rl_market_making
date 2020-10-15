import numpy as np
if __name__ == '__main__':
    import os, sys
    lib_path = os.path.abspath(os.path.join('.'))
    sys.path.append(lib_path)
    from Backtesting.Engine.engine import Market
    from Backtesting.DB_interface.DataReader import dataReader
else:
    from Backtesting.Engine.engine import Market
    from Backtesting.DB_interface.DataReader import dataReader

import pandas as pd


class One_Asset_Env:
    def __init__(self,
                 action_spec,
                 obs_spec,
                 symbol,
                 nTimeStep,
                 # rf,
                 start_date,
                 end_date,
                 n_share,
                 commision = 0.0,
                 isSave = True,
                 ita = 0.01):
        self._action_spec = action_spec
        self._obs_spec = obs_spec
        self.mkt = Market(start_date=start_date,
                 end_date=end_date,
                 cash=1000000,
                          rf = 0,
                          commision= commision)
        self.dr = dataReader()
        self.dr.set_date(start_date)
        self.start_date = start_date
        self.end_date = end_date
        self.nTimeStep = nTimeStep
        # self.rf = rf # risk free rate
        self.counter = 0
        self.done =True
        self.name = 'one_asset'
        self.symbol = symbol
        self.n_share = n_share
        self.obs = None
        self.isSave = isSave

        self._cols = ['BA_spead', 'last_traded_price', 'Smart_price', 'Liquidity_imb', 'market_cost',
                'log_return','reward(log_r)','port_v', 'done']
        self.df = pd.DataFrame(columns=self._cols)
        self.df_idx = 0
        self.cur_pos = 0
        self.A = 1
        self.B = 1
        self.ita = ita

    def step(self, action):
        # if not action.shape == self._action_spec:
        #     raise TypeError('action shape error, suppose to be %s, but input is %s' % (self._action_spec,action.shape))
        if len(action) == 3:
            a_max = action.argmax()
            print(action)
            if a_max == 0:
                realAction = -1
            elif a_max == 1:
                realAction = 0
            else:
                realAction = 1
        elif len(action) == 1:
            print(action)
            realAction = 1 if action[0]>0 else -1
        else:
            raise TypeError(f'Action: {action}')
        print(f'realaction: {self.n_share*realAction}')
        pos = pd.DataFrame([self.n_share*realAction], index = [self.symbol])
        self.mkt.adjustPosition(pos)
        self.cur_pos = self.n_share*realAction
        self.counter += 1
        ret = self.mkt.next()
        self.dr.next()
        if ret == 0:
            self.done = True
        next_obs = self._get_obs()
        p_v = self.mkt.portfolio_account['portfolio_value']
        print(f'portfolio_value: {p_v}')
        reward = np.log(p_v/self.last_value)
        self.last_value = p_v

        # if self.isSave:
        #     # tmp = next_obs[-1].tolist()+[reward, p_v, self.done]
        #     # print('-------------', self.tmp_obs)
        #     self.df.loc[self.df_idx] = tmp
        #     self.df_idx += 1
        print(f'obs: {next_obs}')
        return next_obs, reward,  self.done, dict()

    def diff_sharpe(self, r):
        deltaA = r-self.A
        deltaB = r**2 - self.B

        ret = self.B*deltaA - 0.5 * self.A * deltaB

        self.A = self.ita*r + (1-self.ita)*self.A
        self.B = self.ita*r**2 + (1-self.ita)*self.B

        return ret
    def _get_obs(self):
        tmp = self.dr.get('State', self.nTimeStep)
        tmp = tmp.iloc[:, -1]
        tmp = np.reshape(tmp, newshape=(self.nTimeStep,))
        tmp = np.append(tmp*1000, self.cur_pos/1000)

        return tmp

    def action_space(self):
        return self._action_spec

    def obs_space(self):
        return self._obs_spec

    def reset(self):
        self.mkt.reset()
        self.dr.set_date(self.start_date)
        self.cur_pos = 0
        self.obs = self._get_obs()
        self.counter = 0
        print('env: reset')
        self.done = False
        self.last_value = self.mkt.portfolio_account['portfolio_value']
        return self.obs

    def terminate(self):
        pass

    def save_to_csv(self, epoch):
        try:
            self.df.to_csv(f'./iteration_info/itr_{epoch}.csv')
            self.df = pd.DataFrame(columns=self._cols)
        except FileNotFoundError:
            os.makedirs(f'./iteration_info/', exist_ok= True)
            self.df.to_csv(f'./iteration_info/itr_{epoch}.csv')
            self.df = pd.DataFrame(columns=self._cols)

if __name__ == '__main__':
    env = One_Asset_Env(action_spec= 1,
                        obs_spec=5,
                        symbol='AAPL',
                        nTimeStep=5,
                        start_date=100,
                        end_date=2000,
                        n_share=10)

    obs = env.reset()
    done = False
    action = np.array([1])

    while True:
        if done:
            print('Finished!')
            break
        print(f'obs: {obs}')
        next_obs, reward, done, _ = env.step(action)
        print(f'rwd: {reward}')
        action *= -1
        obs = next_obs


