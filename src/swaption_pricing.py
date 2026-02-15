import numpy as np

class SwaptionPricer:
    def __init__(self, strike, weights, option_type='payer'):
        """
        Initializes the European Swaption pricer.
        
        Parameters:
        strike (float): The strike rate (K) of the swaption.
        weights (numpy.ndarray): 1D array of frozen weights for the forward rates, 
                                 calculated at t=0 based on the yield curve.
        option_type (str): 'payer' (call on the swap rate) or 
                           'receiver' (put on the swap rate).
        """
        self.strike = strike
        self.weights = np.asarray(weights)
        self.option_type = option_type.lower()
        
        if self.option_type not in ['payer', 'receiver']:
            raise ValueError("option_type must be either 'payer' or 'receiver'")

    def calculate_swap_rate(self, simulated_forward_rates):
        """
        Maps the terminal forward overnight rates to the par swap rate.
        
        Parameters:
        simulated_forward_rates (numpy.ndarray): Matrix of shape (n_paths, n_forwards) 
                                                 representing the forward rates at maturity T.
                                                 
        Returns:
        numpy.ndarray: The simulated par swap rates at maturity, shape (n_paths,).
        """
        # The swap rate S_T is a linear combination of the forward rates F_i(T)
        # S_T = sum(w_i * F_i(T))
        # We use numpy dot product to apply the weights across all paths instantly.
        swap_rates_at_maturity = simulated_forward_rates @ self.weights
        
        return swap_rates_at_maturity

    def calculate_payoff(self, swap_rates_at_maturity):
        """
        Calculates the undiscounted payoff of the swaption at maturity.
        
        Parameters:
        swap_rates_at_maturity (numpy.ndarray): Array of simulated swap rates, shape (n_paths,).
        
        Returns:
        numpy.ndarray: The payoff for each path, shape (n_paths,).
        """
        if self.option_type == 'payer':
            # Payer swaption: Right to pay fixed (strike) and receive floating.
            # Payoff = max(S_T - K, 0)
            payoff = np.maximum(swap_rates_at_maturity - self.strike, 0.0)
        else:
            # Receiver swaption: Right to receive fixed (strike) and pay floating.
            # Payoff = max(K - S_T, 0)
            payoff = np.maximum(self.strike - swap_rates_at_maturity, 0.0)
            
        return payoff

# --- Example Usage ---
# strike = 0.045  # 4.5% strike
# # Assume a simple 2-period swap with equal weights for demonstration
# weights = np.array([0.5, 0.5]) 
# 
# pricer = SwaptionPricer(strike, weights, option_type='payer')
# 
# # 'terminal_rates' would be the final column of our simulated F_t paths
# # (assuming we simulated a multi-dimensional F_t for multiple forward tenors)
#