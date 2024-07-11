import numpy as np
import gymnasium as gym
from DataFeature import DataFeature
from utils.AppSetting import AppSetting
import time
import numpy as np


class InsideEnv:
    def __init__(self, bars_count, commission_perc):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.N_steps = 1000  # 這遊戲目前使用多少步學習
        self.setting = AppSetting.get_DTM_setting()

    def _cur_close(self):
        """
            Calculate real close price for the current bar
        """
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open * (1.0 + rel_close)

    def reset(self, prices, offset):
        assert offset >= self.bars_count-1
        self._offset = offset

        # 初始化相關資料
        self._init_cash = 10000

        self._map = {
            "last_action": 0.0,
            "last_share": 0.0,  
            "diff_share": 0.0,
            "calculate_cash": self._init_cash,
            "postion_value": 0.0,
            "commission": 0.0,
            "slippage": 0.0,
            "Close": 0.0,
            "sum": self._init_cash
        }

        self._prices = prices
        self.action = 0

    def step(self, action):
        """
            # 專門為了純多單製作
            計算動態資產分配action 可能的範圍 0 ~ 1

                0.8429519115130819
                0.5133293824170155
                0.3696803100350855
                0.4621302561424804
                0.37277980115611786

            buy and sell
        """

        self.action = action
        reward = 0.0
        done = False

        _cur_close = self._cur_close()

        # 取得即時下單比例
        thisusemoeny = (self._map['last_share'] * _cur_close  + self._map['calculate_cash']) * abs(action)
        # 預估說大概可以購買多少的部位並且不能超過原始本金

        cost = _cur_close * (1 + self.setting['DEFAULT_SLIPPAGE'] + self.commission_perc)
        share = thisusemoeny / cost
        
        self._map['last_action'] = action
        diff_share = share - self._map['last_share']
        self._map['diff_share'] = diff_share

        # 計算傭金和滑價
        commission = abs(diff_share) * _cur_close * self.commission_perc
        slippage = abs(diff_share) * _cur_close * self.setting['DEFAULT_SLIPPAGE']

        # 計算賣出獲得金額 及買入獲得金額
        thissellmoney = abs(self._map['last_share']) * _cur_close
        thisbuymoney = abs(share) * _cur_close
        self._map['calculate_cash'] = self._map['calculate_cash'] + thissellmoney - thisbuymoney - commission  - slippage

        self._map['last_share'] = share
        self._map['commission'] = commission
        self._map['slippage'] = slippage
        self._map['postion_value'] = abs(share) * _cur_close
        self._map['Close'] = _cur_close
        reward = ((self._map['calculate_cash'] + self._map['postion_value']) -self._map['sum']) / self._map['sum']
        self._map['sum'] = self._map['calculate_cash'] + self._map['postion_value']

        # 上一個時步的狀態 ================================
        self._offset += 1
        # 判斷遊戲是否結束
        done |= self._offset >= self._prices.close.shape[0] - 1

        print("step內---------------------------------------------------------")
        print("此次內容:")
        print(self._map)
        time.sleep(5)
        print("step內---------------------------------------------------------")
        return reward, done


    # def step(self, action):
    #     """
    #         計算動態資產分配action 可能的範圍 

    #             0.8429519115130819
    #             -0.5133293824170155
    #             0.3696803100350855
    #             -0.4621302561424804
    #             0.37277980115611786

    #         buy and sell
    #     """
    #     self.action = action
    #     reward = 0.0
    #     done = False

    #     _cur_close = self._cur_close()

    #     # 取得即時下單比例
    #     thisusemoeny = (self._map['postion_value'] + self._map['calculate_cash']) * abs(action)

    #     # 預估說大概可以購買多少的部位並且不能超過原始本金
    #     cost = _cur_close * (1 + self.setting['DEFAULT_SLIPPAGE'] + self.commission_perc)
    #     if np.sign(self._map['last_action']) != np.sign(action) and self._map['last_action'] != 0:
    #         share = (thisusemoeny - self._map['last_share'] * _cur_close * (self.setting['DEFAULT_SLIPPAGE'] + self.commission_perc)) / cost
    #     else:
    #         share = thisusemoeny / cost

    #     # 要考慮正負號 因為部位
    #     share = share * np.sign(action)
    #     self._map['last_action'] = action
    #     diff_share = share - self._map['last_share']
    #     self._map['diff_share'] = diff_share

    #     # 計算傭金和滑價
    #     commission = abs(diff_share) * _cur_close * self.commission_perc
    #     slippage = abs(diff_share) * _cur_close * self.setting['DEFAULT_SLIPPAGE']

    #     # 計算賣出獲得金額 及買入獲得金額
    #     thissellmoney = abs(self._map['last_share']) * _cur_close
    #     thisbuymoney = abs(share) * _cur_close
    #     self._map['calculate_cash'] = self._map['calculate_cash'] + thissellmoney - thisbuymoney - commission  - slippage

    #     self._map['last_share'] = share
    #     self._map['commission'] = commission
    #     self._map['slippage'] = slippage
    #     self._map['postion_value'] = abs(share) * _cur_close
    #     self._map['Close'] = _cur_close

    #     reward = ((self._map['calculate_cash'] + self._map['postion_value']) -self._map['sum']) / self._map['sum']
    #     self._map['sum'] = self._map['calculate_cash'] + self._map['postion_value']

    #     # 上一個時步的狀態 ================================
    #     self._offset += 1
    #     # 判斷遊戲是否結束
    #     done |= self._offset >= self._prices.close.shape[0] - 1

    #     # print("step內---------------------------------------------------------")
    #     # print("此次動作:",action)
    #     # print("此次內容:")
    #     # print(self._map['sum'])
    #     # time.sleep(5)
    #     # print("step內---------------------------------------------------------")
    #     return reward, done


class TimeStep(InsideEnv):

    @property
    def shape(self):
        return (self.bars_count, 5)

    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        ofs = self.bars_count
        for bar_idx in range(self.bars_count):
            res[bar_idx][0] = self._prices.high[self._offset - ofs + bar_idx]
            res[bar_idx][1] = self._prices.low[self._offset - ofs + bar_idx]
            res[bar_idx][2] = self._prices.close[self._offset - ofs + bar_idx]
            res[bar_idx][3] = self._prices.volume[self._offset - ofs + bar_idx]

        res[:, 4] = self.action  # 這邊其實是百分比動作(上一個的)
        return res


class Env:
    def __init__(self, bars_count: int, prices: dict, random_ofs_on_reset):
        self.bars_count = bars_count
        self._count_state = TimeStep(bars_count=self.bars_count,
                                     commission_perc=0.002
                                     )
        self.done = False
        self._prices = prices
        self.get_space()
        self.random_ofs_on_reset = random_ofs_on_reset

    def get_space(self):
        # 定義動作空間和觀察空間
        # 定義連續動作空間
        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32)

        # 假設 self._count_state.shape 是一個 tuple，描述觀察空間的形狀
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._count_state.shape[1],), dtype=np.float32)

    def reset(self):
        self._instrument = np.random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]

        bars = self._count_state.bars_count
        if self.random_ofs_on_reset:
            offset = np.random.choice(prices.high.shape[0]-bars*10) + bars
        else:
            offset = bars
        print("目前步數:", offset)
        self._count_state.reset(prices, offset)
        return self._count_state.encode()

    def step(self, action):
        reward, done = self._count_state.step(action)  # 這邊會更新步數
        obs = self._count_state.encode()  # 呼叫這裡的時候就會取得新的狀態
        info = {
            "instrument": self._instrument,
            "offset": self._count_state._offset,

        }
        return obs, reward, done, info

    def render(self):
        # 簡單打印當前狀態
        pass

    def close(self):
        pass


# app = TimeStep(bars_count=300, commission_perc=0.002)
# app.reset(prices=DataFeature().get_train_net_work_data_by_path(
#     ['BTCUSDT'])['BTCUSDT'], offset=300)
# while True:
#     app.step(action=np.random.uniform(0, 1))
#     time.sleep(1)
