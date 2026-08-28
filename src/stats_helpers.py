import numpy as np
import scipy.stats as stats

def perform_z_proportion_test(size_c, conv_c, size_v, conv_v, alpha):
    """
    Performs a two-tailed Z-proportion hypothesis test.
    Returns statistical metrics and details for evaluation.
    """
    conversions_c = int(size_c * conv_c)
    conversions_v = int(size_v * conv_v)
    
    # Pooled proportion
    p_pool = (conversions_c + conversions_v) / (size_c + size_v)
    
    # Standard error
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / size_c + 1 / size_v))
    
    # Z-statistic
    if se > 0:
        z_stat = (conv_v - conv_c) / se
    else:
        z_stat = 0.0
        
    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    # Critical value (two-tailed)
    critical_z = stats.norm.ppf(1 - alpha / 2)
    
    # Statistical power calculation
    power = 1 - stats.norm.cdf(critical_z - abs(z_stat))
    
    # Significance check
    is_significant = p_value < alpha
    
    # Required sample size per group for 80% power at alpha
    # Standard formula: N = 16 * p * (1 - p) / (effect_size^2)
    effect_size = abs(conv_v - conv_c)
    if effect_size > 0:
        required_sample_size = (16 * p_pool * (1 - p_pool)) / (effect_size ** 2)
    else:
        required_sample_size = float('inf')
        
    return {
        "z_stat": z_stat,
        "p_value": p_value,
        "power": power,
        "is_significant": is_significant,
        "critical_z": critical_z,
        "required_sample_size": required_sample_size
    }

def get_normal_distribution_data(critical_z):
    """
    Generates standard normal distribution curve coordinates
    and definitions of critical rejection regions.
    """
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x)
    
    # Rejection region boundary bounds
    x_left_rejection = np.linspace(-4, -critical_z, 100)
    y_left_rejection = stats.norm.pdf(x_left_rejection)
    
    x_right_rejection = np.linspace(critical_z, 4, 100)
    y_right_rejection = stats.norm.pdf(x_right_rejection)
    
    return {
        "x": x,
        "y": y,
        "x_left_rejection": x_left_rejection,
        "y_left_rejection": y_left_rejection,
        "x_right_rejection": x_right_rejection,
        "y_right_rejection": y_right_rejection
    }
