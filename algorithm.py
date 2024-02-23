import numpy as np
from typing import List, Tuple


class UCB:
    def __init__(self, budget, variance, default_list, max_bid=0.20, div=0.0001, use_variance=False):
        self.budget = budget
        self.variance = variance
        self.default_list = default_list
        
        bid_space_dict = {}
        selection_dict = {}
        reward_dict = {}
        for grade, value in enumerate(default_list):
            default = round(value, 4)
            bid_space_dict[grade] = np.arange(default, max_bid, 0.0001)
            num = len(bid_space_dict[grade])
            selection_dict[grade] = np.zeros(num)
            reward_dict[grade] = np.zeros(num)
        # print(bid_space_dict)
        self.bid_space_dict = bid_space_dict
        self.selection_dict = selection_dict
        self.reward_dict = reward_dict
        self.use_variance = use_variance
            
    def bid(self, grade, amount):
        curr_var = amount ** 2 * self.default_list[grade] * (1 - self.default_list[grade])
        
        if self.budget < amount:
            return 0.21, -1
        elif self.use_variance and self.variance < curr_var:
            return 0.21, -1
        
        selections = self.selection_dict[grade]
        bid_space = self.bid_space_dict[grade]
        rewards = self.reward_dict[grade]
        
        zero_indices = np.where(selections == 0)[0]
        if zero_indices.size > 0:
            mid_index = zero_indices[len(zero_indices) // 2]
            bid = bid_space[mid_index]
            
            # print(f"{bid=}, {mid_index=}")
            return bid, mid_index
        else:
            total_pulls = np.sum(selections)

            average_reward = rewards / selections
            confidence_bound = np.sqrt((2 * np.log(total_pulls)) / selections)
            ucb_values = average_reward + confidence_bound
            # print(ucb_values)

            #bid_index = np.argmax(ucb_values)
            #bid = bid_space[bid_index]
            max_value = np.max(ucb_values)
            indices_of_max = np.where(ucb_values == max_value)[0]
            bid_index = np.random.choice(indices_of_max)



            
            # print(f"{bid=}, {bid_index=}")
            return bid_space[bid_index], bid_index 
        
    def update(self, grade, bid, bid_index, amount, win):
        if bid_index == -1:
            return
        selections = self.selection_dict[grade]
        bid_space = self.bid_space_dict[grade]
        rewards = self.reward_dict[grade]
        zero_indices = np.where(selections == 0)[0]
        default = self.default_list[grade]
        #print(f'{grade=}, {bid=}, {amount=}, {win=}, {default=}')
        if zero_indices.size > 0:
            low = zero_indices[0]
            high = zero_indices[-1]
            # If win, any bid lower than this bid will win
            if win:
                update_indices = range(low, bid_index + 1)
                # import pdb; pdb.set_trace()
                new_reward = (bid_space[update_indices] - self.default_list[grade]) * amount
                rewards[update_indices] += new_reward
            # Since curr bid lost, any bid higher than this will lose
            else:
                update_indices = range(bid_index, high + 1)
            selections[update_indices] += 1
        else:
            if win:
                curr_var = amount ** 2 * self.default_list[grade] * (1 - self.default_list[grade])
                self.budget -= amount
                self.variance -= curr_var
                reward = (bid - self.default_list[grade]) * amount
                rewards[bid_index] += reward
                selections[bid_index] += 1
            else:
                selections[bid_index] += 1

class EGreedy:
    def __init__(self, budget, variance, default_list, max_bid=0.20, div=0.0001, epsilon=0.5, use_variance=False):
        self.budget = budget
        self.variance = variance
        self.default_list = default_list
        self.epsilon = epsilon
        self.use_variance = use_variance
        
        bid_space_dict = {}
        selection_dict = {}
        reward_dict = {}
        for grade, value in enumerate(default_list):
            default = round(value, 4)
            bid_space_dict[grade] = np.arange(default, max_bid, 0.0001)
            num = len(bid_space_dict[grade])
            selection_dict[grade] = np.zeros(num)
            reward_dict[grade] = np.zeros(num)
        self.bid_space_dict = bid_space_dict
        self.selection_dict = selection_dict
        self.reward_dict = reward_dict
            
    def bid(self, grade, amount):
        curr_var = amount ** 2 * self.default_list[grade] * (1 - self.default_list[grade])
        if self.budget < amount:
            return 0.21, -1
        elif self.use_variance and self.variance < curr_var:
            return 0.21, -1
        selections = self.selection_dict[grade]
        bid_space = self.bid_space_dict[grade]
        rewards = self.reward_dict[grade]
        
        zero_indices = np.where(selections == 0)[0]
        if zero_indices.size > 0:
            mid_index = zero_indices[len(zero_indices) // 2]
            bid = bid_space[mid_index]
            # print(f"{bid=}, {mid_index=}")
            return bid, mid_index
        else:
            # total_pulls = np.sum(selections)
            average_reward = rewards / selections
            
            dice = np.random.uniform(0, 1)
            if dice <= self.epsilon: # Greedy
                max_value = np.max(average_reward)
                indices_of_max = np.where(average_reward == max_value)[0]
                bid_index = np.random.choice(indices_of_max)
                
            else: # Explore
                # default_index = self.bid_space[np.where(self.bid_space > self.default_list[grade])[0]]
                # import pdb; pdb.set_trace()
                bid_index = np.random.choice(len(selections))
            
            bid = bid_space[bid_index]
            # print(f"{bid=}, {bid_index=}")
            return bid_space[bid_index], bid_index 
        
    def update(self, grade, bid, bid_index, amount, win):
        if bid_index == -1:
            return
        selections = self.selection_dict[grade]
        bid_space = self.bid_space_dict[grade]
        rewards = self.reward_dict[grade]
        zero_indices = np.where(selections == 0)[0]
        if zero_indices.size > 0:
            low = zero_indices[0]
            high = zero_indices[-1]
            # If win, any bid lower than this bid will win
            if win:
                update_indices = range(low, bid_index + 1)
                # import pdb; pdb.set_trace()
                new_reward = (bid_space[update_indices] - self.default_list[grade]) * amount
                rewards[update_indices] += new_reward
            # Since curr bid lost, any bid higher than this will lose
            else:
                update_indices = range(bid_index, high + 1)
            selections[update_indices] += 1
        else:
            if win:
                curr_var = amount ** 2 * self.default_list[grade] * (1 - self.default_list[grade])
                self.budget -= amount
                self.variance -= curr_var
                count = selections[bid_index] 
                value = rewards[bid_index]
                
                reward = (bid - self.default_list[grade]) * amount
                new_value = ((count - 1) / count) * value + (1 / count) * reward
                rewards[bid_index] = new_value
                selections[bid_index] += 1
            else:
                selections[bid_index] += 1
     

class FixedBandit:
    def __init__(self, budget, variance, default_list: np.array, alpha: float, beta_mean: float, use_variance=False):
        self.default_list = default_list
        self.alpha = alpha
        self.beta_mean = beta_mean
        self.budget = budget
        self.variance = variance
        self.bid_mapping = {}
        self.use_variance = use_variance
        for i, v in enumerate(default_list):
            self.bid_mapping[i] = self.sample_from_beta(alpha, beta_mean) * v + v
        
    def bid(self, grade: int, amount: int) -> float:
        curr_var = amount ** 2 * self.default_list[grade] * (1 - self.default_list[grade])
        if self.budget < amount:
            return 0.21, -1
        elif self.use_variance and self.variance < curr_var:
            return 0.21, -1
        return self.bid_mapping[grade], 0
        
    def update(self, amount, grade, win):
        if win:
            curr_var = amount ** 2 * self.default_list[grade] * (1 - self.default_list[grade])
            self.budget -= amount
            self.variance -= curr_var
    
    def sample_from_beta(self, alpha, mean):
        beta = (alpha - mean * alpha) / mean
        return np.random.beta(alpha, beta)