import pandas as pd
from collections import defaultdict
from typing import List
from online_algorithm import BanditFeedbackGradientDescent
from algorithm import UCB, FixedBandit, EGreedy
from auction import Auction
import numpy as np
from tqdm import tqdm



def get_bank_bids(auction, row: pd.Series, budget_list: List[float]) -> List[float]:
    bank_bid_list = []
    original_bid_list = []
    # import pdb; pdb.set_trace()
    for i in range(len(budget_list)):
        # TODO: div by 100 if lc data
        bid = row[f'p{i}_INTEREST']
        
        if f'p{i}_LIMIT' in row.index:
            amount = row[f'p{i}_LIMIT']
        else:
            amount = row[f'AMOUNT']
        if amount > 0 and bid > 0 and bid < 0.2 and budget_list[i] >= amount:
            bank_bid_list.append(bid)
        else:
            bank_bid_list.append(0.21)
        original_bid_list.append((budget_list[i], bid))
    bank_res = auction.get_winner(bank_bid_list)
    if bank_res['winner'] is None: # all bids are over 0.21. So either they all don't have money or the bid is > 0.20
        return None, 0.21, None, [0.21 for x in range(len(budget_list))]
        
    # bank_winner_default_proba = row[f'p{bank_res["winner"]}_DEFAULT_PROBA']
    bank_winner_default_proba = 0
    
    return bank_res['winner'], bank_res['winner_bid'], bank_winner_default_proba, bank_bid_list


def simulate(merged_df, params, num_banks, agent_type='online', with_confidence=False, grade_column='xgb_grade', debug_steps=None):
    if debug_steps is not None:
        print("WARNING: DEBUG MODE")
    auction = Auction()
    budget = int(params['budget_ratio'] * params['total_time'])
    variance = int(params['variance_ratio'] * params['total_time'])
    default_list = [round(x, 4) for x in params['proba_default_list']]
    lambda1_list = []
    lambda2_list = []
    # print(budget)

    if agent_type == 'online':
        agent = BanditFeedbackGradientDescent(
                    params['budget_ratio'], 
                    params['variance_ratio'], 
                    params['epsilon1'], 
                    params['epsilon2'], 
                    params['total_time'], 
                    default_list, 
                    use_variance=params.get('use_variance', False))
    elif agent_type == 'ucb':
        agent = UCB(budget, variance, default_list, use_variance=params.get('use_variance', False))
    elif agent_type == 'egreedy':
        agent = EGreedy(budget, variance, default_list, epsilon=params['e-greedy'], use_variance=params.get('use_variance', False))
    elif agent_type == 'fixed':
        agent = FixedBandit(budget, variance, default_list, alpha=0.1, beta_mean=0.1, use_variance=params.get('use_variance', False))
    else:
        raise ValueError(f"{agent_type} is not supported")
        
    reward_dict = defaultdict(list)
    reward_dict['algo'] = []
    budget_list = [params['budget_ratio'] * params['total_time']] * num_banks

    meta = {}
    algo_bid_dict = defaultdict(list)
    algo_proba_list = []
    algo_budget_list = []
    algo_variance_list = []
    algo_bid_index_list = []
    bank_bid_dict = defaultdict(list)
    bank_budget_dict = defaultdict(list)
    win_list = []
    grade_list = []
    algo_bid_list = []
    lambda2_delta_list = [] 
    # for step, row in tqdm(merged_df.reset_index(drop=True).iterrows()):
    for step, row in merged_df.reset_index(drop=True).iterrows():
        # First run on the banks
        bank_winner, bank_winner_bid, bank_winner_default_proba, bank_bid_list = get_bank_bids(auction, row, budget_list)
        for i, bid_value in enumerate(bank_bid_list):
            bank_bid_dict[i].append(bid_value)

        if 'GRADE' in row.index:
            grade = int(row['GRADE'] - 1)
        else:
            grade = int(row[grade_column] - 1)
        amount = row['AMOUNT']
        # print(row)
        
        # Then run on algo vs banks
        estimated_proba_select = 0
        if agent_type == 'full_feedback':
            algo_bid, estimated_proba_select, bid_index = agent.bid(grade, amount)
        elif agent_type == 'online':
            algo_bid, estimated_proba_select, bid_index = agent.bid(grade, amount, with_confidence=with_confidence)
        elif agent_type in ['ucb', 'egreedy', 'fixed']:
            algo_bid, bid_index = agent.bid(grade, amount)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        # algo_bid = round(algo_bid, 4)
        algo_bid_dict[grade].append(algo_bid)
        algo_proba_list.append(estimated_proba_select)
        algo_bid_index_list.append(bid_index)
        algo_budget_list.append(agent.budget)
        algo_variance_list.append(agent.variance)
        algo_bid_list.append(algo_bid)
        all_bids = [algo_bid] + bank_bid_list
        grade_list.append(grade)
        # print(f"{step=} {grade=}, {} {[round(x, 4) for x in all_bids]}")

        res = auction.get_winner(all_bids)
        win = False

        # print(f"{step=}, {res['winner']=}, {grade=}, {[round(x, 4) for x in all_bids]}")
        default_rate = default_list[grade]
        if res['winner'] == 0: # agent wins
            win = True
            # print(f"{grade=}, {algo_bid=}, {default_rate=}, {bank_bid_list}")
            reward_dict['algo'].append((algo_bid - default_rate) * amount)
            for j in range(len(bank_bid_list)):
                reward_dict[f'p{j}'].append(0)
                bank_budget_dict[f'p{j}'].append(budget_list[j])
        elif res['winner'] is None: # No winner
            win = False
            reward_dict['algo'].append(0)
            for j in range(len(bank_bid_list)):
                reward_dict[f'p{j}'].append(0)
                bank_budget_dict[f'p{j}'].append(budget_list[j])
        elif bank_winner is not None: # bank wins
            budget_list[bank_winner] -= amount
            reward_dict['algo'].append(0)
            for j in range(len(bank_bid_list)):
                if j == bank_winner:
                    reward_dict[f'p{bank_winner}'].append((bank_winner_bid - bank_winner_default_proba) * amount)
                else:
                    reward_dict[f'p{j}'].append(0)
                bank_budget_dict[f'p{j}'].append(budget_list[j])
        win_list.append(win)
        if agent_type == 'online':
            old_lambda2 = agent.lambda2
        if agent_type in ['full_feedback', 'online']:
            agent.update(grade, amount, bank_winner_bid, algo_bid, estimated_proba_select, win)
        elif agent_type in ['ucb', 'egreedy']:
            agent.update(grade, algo_bid, bid_index, amount, win)
        elif agent_type in ['fixed']:
            agent.update(amount, grade, win)
        
        if agent_type == 'online':
            new_lambda2 = agent.lambda2
            lambda2_delta = new_lambda2 - old_lambda2 
            lambda1_list.append(agent.lambda1)
            lambda2_list.append(agent.lambda2)
            lambda2_delta_list.append(lambda2_delta)
            
        if debug_steps is not None:
            if step >= debug_steps:
                break

    
    # meta['testing'] = agent.other_bank[grade]
    meta['grade_list'] = grade_list
    meta['algo_bid_dict'] = algo_bid_dict
    meta['algo_proba_list'] = algo_proba_list
    meta['algo_budget_list'] = algo_budget_list
    meta['algo_variance_list'] = algo_variance_list
    meta['algo_bid_index_list'] = algo_bid_index_list
    meta['bank_bid_dict'] = bank_bid_dict
    meta['bank_budget_dict'] = bank_budget_dict
    meta['win_list'] = win_list
    meta['algo_bid_list'] = algo_bid_list
    meta['lambda1_list'] = lambda1_list
    meta['lambda2_list'] = lambda2_list
    meta['lambda2_delta_list'] = lambda2_delta_list
    return reward_dict, meta

def simulate2(merged_df, params, num_banks, agent_type='full_feedback', with_confidence=False):

    auction = Auction()

    agent1 = BanditFeedbackGradientDescent(**params)
    agent2 = BanditFeedbackGradientDescent(**params)
    
    reward_dict = defaultdict(list)
    reward_dict['algo1'] = []
    reward_dict['algo2'] = []
    meta = {}
    
    for step, row in tqdm(merged_df.iterrows()):
        grade = row['GRADE'] - 1
        amount = row['AMOUNT']
        
        # Then run on algo vs banks
        
        algo_bid1, estimated_proba_select1, bid_index1 = agent1.bid(grade, amount)
        algo_bid2, estimated_proba_select2, bid_index2 = agent2.bid(grade, amount)

        all_bids = [algo_bid1, algo_bid2]

        res = auction.get_winner(all_bids)
        win1 = False
        win2 = False
        default_rate = params['proba_default_list'][grade]
        if res['winner'] == 0:
            win1 = True
            reward_dict['algo1'].append((algo_bid1 - default_rate) * amount)
            reward_dict['algo2'].append(0)
            
        elif res['winner'] == 1:
            win2 = True
            reward_dict['algo1'].append(0)
            reward_dict['algo2'].append((algo_bid2 - default_rate) * amount)

        agent1.update(grade, amount, algo_bid2, algo_bid1, estimated_proba_select1, win1)
        agent2.update(grade, amount, algo_bid1, algo_bid2, estimated_proba_select2, win2)  
        
    return reward_dict, meta