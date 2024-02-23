import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool
from scipy.stats import rankdata
from typing import List

from data import get_random_bank_df
from simulation import simulate


NUM_PROCESS = 50


def run_simulation(args):
    params, credit_df, seed, shuffle, use_variance, confidence = args
    
    main_seed = seed
    if shuffle:
        main_seed = 0
    
    merged_df = get_random_bank_df(credit_df, seed=main_seed, alpha=0.5)

    xgb_default_list = merged_df.groupby('xgb_grade')['y'].mean().values
    xgb_default_list = (xgb_default_list[xgb_default_list < 0.2]).tolist()
    
    rf_default_list = merged_df.groupby('rf_grade')['y'].mean().values
    rf_default_list = (rf_default_list[rf_default_list < 0.2]).tolist()
    max_grade = min(len(xgb_default_list), len(rf_default_list))

    grade_column = 'xgb_grade'
    target_column = 'y'
    if shuffle:
        merged_df = merged_df.sample(frac=1, random_state=seed)

    
    default_list = merged_df.groupby(grade_column)[target_column].mean()
    tmp = merged_df.groupby(grade_column)[target_column].mean().reset_index()
    tmp = tmp.rename(columns={target_column: 'MEAN_PROBA'})
    ff = pd.merge(merged_df, tmp, on=grade_column, how='left')
    max_grade = default_list[default_list < 0.2].index[-1]
    merged_df = ff.loc[ff[grade_column] <= max_grade]
    default_list = default_list.tolist()[:max_grade]

    mean_variance = (ff['AMOUNT'] * ff['AMOUNT'] * ff['MEAN_PROBA'] * (1 - ff['MEAN_PROBA'])).sum() / len(ff)

        
    mean_budget = merged_df['AMOUNT'].mean()

    epsilon2 = 0
    if use_variance:
        epsilon2 = params.get('epsilon2', 2 * 10 ** (-8))
    simulation_params = {
        'budget_ratio': mean_budget * params['budget_multiple'],
        'variance_ratio': mean_variance * params['variance_multiple'],        
        'epsilon1': params.get('epsilon1', 2 * 10 ** (-6)),
        'epsilon2': epsilon2,
        'total_time': len(merged_df),
        'e-greedy': params.get('e-greedy', 0.5),
        'proba_default_list': default_list,
        'use_variance': use_variance
    }
    online_res, online_meta = simulate(merged_df, simulation_params, 10, agent_type='online', with_confidence=confidence)
    ucb_res, ucb_meta = simulate(merged_df, simulation_params, 10, agent_type='ucb', with_confidence=confidence)
    greedy_res, greedy_meta = simulate(merged_df, simulation_params, 10, agent_type='egreedy', with_confidence=confidence)
    fixed_res, fixed_meta = simulate(merged_df, simulation_params, 10, agent_type='fixed', with_confidence=confidence)

    results = {'rewards': {}, 'meta': {}}
    results['rewards']['online'] = online_res
    results['rewards']['ucb'] = ucb_res
    results['rewards']['greedy'] = greedy_res
    results['rewards']['fixed'] = fixed_res

    results['meta']['online'] = online_meta
    results['meta']['ucb'] = ucb_meta
    results['meta']['greedy'] = greedy_meta
    results['meta']['fixed'] = fixed_meta
    results['df'] = merged_df
    return results

def get_reward_stats(results: List):
    online_rewards = [np.cumsum(results[x]['rewards']['online']['algo'])[-1] for x in range(len(results))]
    ucb_rewards = [np.cumsum(results[x]['rewards']['ucb']['algo'])[-1] for x in range(len(results))]
    greedy_rewards = [np.cumsum(results[x]['rewards']['greedy']['algo'])[-1] for x in range(len(results))]
    fixed_rewards = [np.cumsum(results[x]['rewards']['fixed']['algo'])[-1] for x in range(len(results))]

    online_mean = np.mean(online_rewards)
    online_std = np.std(online_rewards)
    
    ucb_mean = np.mean(ucb_rewards)
    ucb_std = np.std(ucb_rewards)

    greedy_mean = np.mean(greedy_rewards)
    greedy_std = np.std(greedy_rewards)

    fixed_mean = np.mean(fixed_rewards)
    fixed_std = np.std(fixed_rewards)

    print(f"AutoInterest: mean={online_mean}\tstd={online_std}")
    print(f"UCB: mean={ucb_mean}\tstd={ucb_std}")
    print(f"E-Greedy: mean={greedy_mean}\tstd={greedy_std}")
    print(f"Fixed: mean={fixed_mean}\tstd={fixed_std}")
    
def get_rank_stats(results: List):

    online_rewards = [np.cumsum(results[x]['rewards']['online']['algo'])[-1] for x in range(len(results))]
    ucb_rewards = [np.cumsum(results[x]['rewards']['ucb']['algo'])[-1] for x in range(len(results))]
    greedy_rewards = [np.cumsum(results[x]['rewards']['greedy']['algo'])[-1] for x in range(len(results))]
    fixed_rewards = [np.cumsum(results[x]['rewards']['fixed']['algo'])[-1] for x in range(len(results))]

    zipped_arr = np.array(list(zip(online_rewards, ucb_rewards, greedy_rewards, fixed_rewards)))
    rank_array = rankdata(-zipped_arr, axis=1)
    ranks = np.mean(rank_array, axis=0)

    online_rank = ranks[0]
    ucb_rank = ranks[1]
    greedy_rank = ranks[2]
    fixed_rank = ranks[3]

    print(f"AutoInterest rank:{online_rank}")
    print(f"UCB rank:{ucb_rank}")
    print(f"E-Greedy rank:{greedy_rank}")
    print(f"Fixed rank:{fixed_rank}")


def plot_single_stats(res: List):
    for algo_type in ['online', 'ucb', 'greedy', 'fixed']:
        reward_list = np.cumsum(res[0]['rewards'][algo_type]['algo'])
        budget_list = res[0]['meta'][algo_type]['algo_budget_list']
        var_list = res[0]['meta'][algo_type]['algo_variance_list']
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        axes[0].plot(range(len(reward_list)), reward_list)
        axes[1].plot(range(len(reward_list)), budget_list)
        axes[2].plot(range(len(reward_list)), var_list)
        axes[0].set_title(f"{algo_type} Cummulative Reward")
        axes[1].set_title(f"{algo_type} Budget")
        axes[2].set_title(f"{algo_type} Variance")
        plt.show()
       
def plot_lambda(res):
    lambda1_list = res['meta']['online']['lambda1_list']
    lambda2_list = res['meta']['online']['lambda2_list']
    lambda2_delta_list = res['meta']['online']['lambda2_delta_list']
    _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].plot(range(len(lambda1_list)), lambda1_list)
    axes[0].set_title('lambda1')
    axes[1].plot(range(len(lambda2_list)), lambda2_list)
    axes[1].set_title('lambda2')
    axes[2].plot(range(len(lambda2_delta_list)), lambda2_delta_list)
    axes[2].set_title('lambda2 Delta')
    plt.show()


def run_epsilon(credit_df, use_variance, confidence, budget_multiple, variance_multiple, epsilon_list, epsilon_str_list, shuffle, epsilon_type='epsilon1', seed=0, color_list=None):  
    with Pool(processes=10) as pool:
        fig, axes = plt.subplots(1, 1, figsize=(5, 5))
        for i, dataset_type in enumerate(['credit_default']):

            if epsilon_type == 'epsilon1':
                args_list = [({'budget_multiple': budget_multiple, 'variance_multiple': variance_multiple, 'epsilon1': epsilon_list[index]}, credit_df, seed, shuffle, use_variance, confidence) for index in range(len(color_list))]
            else:
                args_list = [({'budget_multiple': budget_multiple, 'variance_multiple': variance_multiple, 'epsilon2': epsilon_list[index]}, credit_df, seed, shuffle, use_variance, confidence) for index in range(len(color_list))]
    
            results = pool.map(run_simulation, args_list)
        
            for j, rewards in enumerate(results):
                online_rewards = np.cumsum(rewards['rewards']['online']['algo'])
                axes.plot(online_rewards, label=f'ε={epsilon_str_list[j]}', color=color_list[j])
    
            axes.set_xlabel("Round")
            axes.set_title(dataset_type)
            if i == 0:
                axes.set_ylabel("Cummulative Reward")
                if epsilon_type == 'epsilon1':
                    axes.legend(loc='upper left')
                else:
                    axes.legend(loc='lower right')
            else:
                axes.legend(loc='lower right')
    
    filename = f'images/merged_{epsilon_type}_{budget_multiple=}_{variance_multiple=}'
    plt.savefig(f'{filename}.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(f'{filename}.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()

def run_single(credit_df, shuffle, use_variance, confidence, epsilon_list, budget_multiple, variance_multiple):
    for i, epsilon in enumerate(epsilon_list):
        print("*"*80)
        print(epsilon_list[i])
        

        res = run_simulation([({'budget_multiple': budget_multiple, 'variance_multiple': variance_multiple, 'epsilon2': epsilon}, credit_df, seed, shuffle, use_variance, confidence) for seed in range(100)][i])

        plot_single_stats([res])
        plot_lambda(res)


def run_parallel_budget(credit_df, multiple_list, shuffle, use_variance, use_confidence):
    variance_multiple = 1
    with Pool(processes=NUM_PROCESS) as pool:
        for multiple in multiple_list:
            print(f"{multiple=:.2f}, {shuffle=}, {use_variance=}, {use_confidence=}")
            
            args_list = [({'budget_multiple': multiple, 'variance_multiple': variance_multiple}, credit_df, seed, shuffle, use_variance, use_confidence) for seed in range(100)]
        
            results = pool.map(run_simulation, args_list)
            if shuffle:
                get_reward_stats(results)
            else:
                get_rank_stats(results)
            # plot_single_stats(results)
            print("*"*80)
        
            
def run_parallel_variance(credit_df, multiple_list, shuffle, use_variance, use_confidence):
    budget_multiple = 1
    with Pool(processes=NUM_PROCESS) as pool:
        for multiple in multiple_list:
            print(f"{multiple=:.2f}, {shuffle=}, {use_variance=}, {use_confidence=}")
            
            args_list = [({'budget_multiple': budget_multiple, 'variance_multiple': multiple}, credit_df, seed, shuffle, use_variance, use_confidence) for seed in range(100)]
        
            results = pool.map(run_simulation, args_list)
            if shuffle:
                get_reward_stats(results)
            else:
                get_rank_stats(results)
            plot_single_stats(results)
            print("*"*80)

def run_budget_change(credit_df, use_variance, use_confidence, seed=0, color_list=['#d7191c','#fdae61','#abd9e9','#2c7bb6']):  
    agent_list = ['online', 'ucb', 'greedy', 'fixed']
    
    budget_multiple_list = [0.1, 0.2, 0.3]
    shuffle = True
    
    i, axes = plt.subplots(1, 3, figsize=(30, 6))
    for i, budget_multiple in enumerate(budget_multiple_list):
        ax = axes[i]
        # def run_simulation(args):
        #     params, credit_df, seed, shuffle, use_variance, confidence = args
        res = run_simulation([({'budget_multiple': budget_multiple, 'variance_multiple': 1}, credit_df, seed, shuffle, use_variance, use_confidence) for _ in range(100)][0])
        reward_dict = res['rewards']
        
        for color_index, agent_type in enumerate(agent_list):
            if agent_type == 'online':
                label = 'AutoInterest'
            else:
                label = agent_type
            ax.plot(np.cumsum(reward_dict[agent_type]['algo']), label=label, color=color_list[color_index])

        dataset_name = 'Credit Default'
        title = f"{dataset_name}, Budget Multiplier: {round(budget_multiple, 2)}"
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.set_ylabel("Cumulative Reward")
        ax.legend()          
    plt.savefig(f'images/cumulative_reward_{seed=}.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(f'images/cumulative_reward_{seed=}.png', format='png', dpi=300, bbox_inches='tight')
    plt.show()
         

def run_ranking_experiment(credit_df, multiple_list, use_variance=False, use_confidence=True):
    run_parallel_budget(credit_df, multiple_list, False, use_variance, use_confidence)
    
    
def run_reward_experiment(credit_df, multiple_list, use_variance=False, use_confidence=True):
    run_parallel_budget(credit_df, multiple_list, True, use_variance, use_confidence)
    
    
def bid_experiment(credit_df, budget_multiple, use_confidence, seed=0):
    variance_multiple = 1
    use_variance = False
    res = run_simulation([({'budget_multiple': budget_multiple, 'variance_multiple': variance_multiple}, credit_df, seed, False, use_variance, use_confidence) for _ in range(100)][0])
    return res

def run_bid_experiment(credit_df, budget_multiple1, budget_multiple2, use_confidence=True, grade=5, seed=0):
    res = bid_experiment(credit_df, budget_multiple1, use_confidence, seed=seed)
    res2 = bid_experiment(credit_df, budget_multiple2, use_confidence, seed=seed)

    bids = res['meta']['online']['algo_bid_dict'][grade]
    bids = [x for x in bids if x < 0.21]
    
    bids2 = res2['meta']['online']['algo_bid_dict'][grade]
    bids2 = [x for x in bids2 if x < 0.21]

    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(bids)
    axes[0].set_title(f"Credit Default, Budget Multiplier: {budget_multiple1:.2f}, Grade: {grade+1}")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Interest Rate")
    axes[1].plot(bids2)
    axes[1].set_title(f"Credit Default, Budget Multiplier: {budget_multiple2:.2f}, Grade: {grade+1}")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Interest Rate")
    plt.show()


    
def run_budget_epsilon(credit_df, budget_multiple, variance_multiple, epsilon_list, epsilon_str_list, seed=0):
    color_list = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c']

    use_variance = False
    confidence = True
    shuffle = False
    epsilon_type = 'epsilon1'

    run_epsilon(credit_df, use_variance, confidence, budget_multiple, variance_multiple, epsilon_list, epsilon_str_list, shuffle, epsilon_type=epsilon_type, seed=seed, color_list=color_list)


def run_variance_epsilon(credit_df, budget_multiple, variance_multiple, epsilon_list, epsilon_str_list, seed=0):
    color_list = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c']

    use_variance = True
    confidence = True
    shuffle = False
    epsilon_type = 'epsilon2'

    run_epsilon(credit_df, use_variance, confidence, budget_multiple, variance_multiple, epsilon_list, epsilon_str_list, shuffle, epsilon_type=epsilon_type, seed=seed, color_list=color_list)
