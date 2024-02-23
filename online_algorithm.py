import numpy as np
# online algorithm

# Gradient Descent Algorithm when we observe other bank's strategies each round

class BanditFeedbackGradientDescent:
    def __init__(self, budget_ratio, variance_ratio, epsilon1, epsilon2, total_time, proba_default_list, max_bid=0.20, use_variance=False, **kwargs):
        
        self.budget_ratio = budget_ratio
        self.variance_ratio = variance_ratio
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.lambda1 = 0
        self.lambda2 = 0
        self.max_bid = max_bid
        self.bid_dict = {}
        self.num_grade = len(proba_default_list)
        proba_default_list = [round(x, 4) for x in proba_default_list]
        self.proba_default_list = proba_default_list
        self.use_variance = use_variance
        
        for i in range(self.num_grade) :
            # self.bid_dict[i] = np.linspace(proba_default_list[i], max_bid, num=int(max_bid/0.0001)+1)
            self.bid_dict[i] = np.arange(proba_default_list[i], max_bid, 0.0001)
        # print(self.bid_dict)

        self.time_space = [0 for i in range(self.num_grade)]
        self.total_time = total_time
        self.grade_bid_space_count = {}
        self.grade_bid_space_reward = {}
        for i in range(self.num_grade):
            bid_space = self.bid_dict[i]
            self.grade_bid_space_count[i] = np.zeros(len(bid_space))
            self.grade_bid_space_reward[i] = np.zeros(len(bid_space))
        self.budget = budget_ratio * total_time
        self.variance = variance_ratio * total_time

    def bid(self, grade, amount, with_confidence=False):
        bid_space = self.bid_dict[grade]
        if self.use_variance:
            curr_var = amount ** 2 * self.proba_default_list[grade] * (1 - self.proba_default_list[grade])
            if self.budget < amount or self.variance < curr_var: 
                return 0.21, 0, 0
        else:
            if self.budget < amount:
                return 0.21, 0, 0
            
        current_count = self.grade_bid_space_count[grade]
        current_reward = self.grade_bid_space_reward[grade]
        zero_loc = np.where(current_count == 0)[0]
        proba_default = self.proba_default_list[grade]

        if len(zero_loc) > 0:
            ind = zero_loc[0] + int((zero_loc[-1]-zero_loc[0])/2)
            return bid_space[ind], 0, ind
        else:
            current_proba = current_reward / current_count
            delta = (np.array(range(len(bid_space))) +1) / len(bid_space)
            paced_grade = np.log((self.time_space[grade] + 1) * delta)
            
            paced_grade = np.where(paced_grade > 0, paced_grade, 0)
            confidence = np.sqrt(paced_grade / current_count)

            if with_confidence:
                bid_array = (bid_space * amount
                             - proba_default * amount 
                             - self.lambda1 * amount
                             - self.lambda2 * amount ** 2 * proba_default * (1 - proba_default)) * current_proba + confidence * amount * current_proba
            else:
                bid_array = (bid_space * amount
                             - proba_default * amount
                             - self.lambda1 * amount 
                             - self.lambda2 * amount ** 2 * proba_default * (1 - proba_default)) * current_proba
            best_bid_index = np.argmax(bid_array)
            return bid_space[best_bid_index], current_proba[best_bid_index], best_bid_index

    def update(self, grade, amount, other_bank, bid, estimated_proba_select, win):
        bid_space = self.bid_dict[grade]
        self.time_space[grade] += 1
        proba_default = self.proba_default_list[grade]
        if bid > self.max_bid:
            return

        curr_var = amount ** 2 * self.proba_default_list[grade] * (1 - self.proba_default_list[grade])
        # print(f"{curr_var=:.4f}, {self.lambda2:.10f}")
        # print(f"\t{self.variance_ratio=:.4f},  {amount ** 2 * proba_default * (1 - proba_default) * estimated_proba_select}, {estimated_proba_select=:.4f}")
        if win:
            self.grade_bid_space_count[grade][np.where(bid_space <= bid)[0]] += 1
            self.grade_bid_space_reward[grade][np.where(bid_space <= bid)[0]] += 1
            self.budget -= amount
            self.variance -= curr_var
            self.lambda1 -= self.epsilon1 * (self.budget_ratio - amount * estimated_proba_select)
            self.lambda1 = max(0, self.lambda1)
            self.lambda2 -= self.epsilon2 * (self.variance_ratio - amount ** 2 * proba_default * (1 - proba_default) * estimated_proba_select)
            # print(f"new labmda2-1: {self.lambda2}")
            self.lambda2 = max(0, self.lambda2)
            # print(f"new lambda1: {self.lambda1}")
            # print(f"new lambda2: {self.lambda2}")
        else:
            self.grade_bid_space_count[grade][np.where(bid_space >= bid)[0]] += 1
            self.lambda1 -= self.epsilon1 * (self.budget_ratio - amount * estimated_proba_select)
            self.lambda1 = max(0, self.lambda1)
            self.lambda2 -= self.epsilon2 * (self.variance_ratio - amount ** 2 * proba_default * (1 - proba_default) * estimated_proba_select)
            self.lambda2 = max(0, self.lambda2)
            # print(f"\tnew lambda1: {self.lambda1}")
            # print(f"\tnew lambda2: {self.lambda2}")
            