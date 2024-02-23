from typing import List, Dict, Union, Tuple
import pandas as pd
import numpy as np
import random
from tqdm import tqdm


class Auction:
    def get_winner(self,
                   bid_list: List[float],
                  ) -> Dict[str, float]:
        bid_list = np.array(bid_list)
        valid_indices = np.where((bid_list > 0) & (bid_list <= 0.2))[0]
    
        # If there are valid indices, proceed
        if valid_indices.size > 0:
            # Extracting values for valid indices
            valid_values = bid_list[valid_indices]
        
            # Finding the maximum value among the valid values
            min_value = np.min(valid_values)
        
            # Finding all indices where the maximum value occurs
            min_value_indices = valid_indices[valid_values == min_value]
        
            # Selecting a random index if there's a tie
            min_index_among_valid = random.choice(min_value_indices)
            return {'winner': min_index_among_valid,
                'winner_bid': bid_list[min_index_among_valid]}
        else:
            min_index_among_valid = None
            return {'winner': min_index_among_valid,
                    'winner_bid': 0.21}
        

