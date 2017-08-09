import tensorflow as tf

def get_line_eq(x_points, y_points):
    """
    Returns spline appromation (slope & y-intercept) for given x and y values
    Inputs:
      x_points [1D tf.tensor] of len [num_bins] containing histogram bin edges
      y_points [1D tf.tensor] of len [num_bins] containing histogram counts
    Outputs:
      tuple containing (ms, bs) evaluated for given pairs of points ((x_i, y_i), (x_i+1, y_i+1))
        the last index of ms and bs is concatenated onto itself, such that ms[-1] == ms[-2] and bs[-1] == bs[-2]
    """
    # Concats add last line again, so len(ms) == len(bs) == len(xs) == len(ys)
    # y[-1] = m[-1] x[-1] + b[-1] ; y[1] = m[0]x[1]+b[0] = m[1]x[1]+b[1]
    ms = tf.divide(tf.subtract(y_points[1:], y_points[:-1]), tf.subtract(x_points[1:], x_points[:-1]))
    last_m = tf.divide(tf.subtract(y_points[-1], y_points[-2]), tf.subtract(x_points[-1], x_points[-2]))
    ms = tf.concat([ms, tf.expand_dims(last_m, axis=0)], axis=0)
    bs = tf.subtract(y_points, tf.multiply(ms, x_points))
    return (ms, bs)

def integral_end_point(x_points, y_points, ms, bs):
    """
    Compute the integral end point for spline entropies evaluated at points specified by inputs
    Inputs:
      x_points [1D tf.tensor] of len [num_bins] containing histogram bin edges
      y_points [1D tf.tensor] of len [num_bins] containing histogram counts
      ms [1D tf.tensor] of len [num_bins] containing line slopes
        The last entry is copied twice, such that ms[-2] == ms[-1]
      bs [1D tf.tensor] of len [num_bins] containing line y-intercepts
        The last entry is copied twice, such that bs[-2] == bs[-1]
    Outputs:
      end_points [1D tf.tensor] of len [num_bins] integral evaluated from 0 to each value specified in x_points
    """
    # - integral_{0, x0}(mx+b * log(mx+b)
    # = (mx(mx+2b) - 2*(mx+b)^2*log(mx+b))/4m | {x0}
    m_x = tf.multiply(x_points, ms)
    left_side = tf.multiply(m_x, tf.add(m_x, tf.multiply(2.0, bs)))
    right_side = tf.multiply(tf.multiply(2.0, tf.square(y_points)), tf.log(y_points))
    end_points = tf.divide(tf.subtract(left_side, right_side), tf.multiply(4.0, ms))
    return end_points

def get_differential_entropy(x_points, y_points, ms, bs):
    """
    Return differential entropy for splines computed between all histogram points
    Inputs:
      x_points [1D tf.tensor] of len [num_bins] containing histogram bin edges
      y_points [1D tf.tensor] of len [num_bins] containing histogram counts
      ms [1D tf.tensor] of len [num_bins] containing line slopes
        The last entry is copied twice, such that ms[-2] == ms[-1]
      bs [1D tf.tensor] of len [num_bins] containing line y-intercepts
        The last entry is copied twice, such that bs[-2] == bs[-1]
    Outputs:
      diff_entropy [1D tf.tensor] of len [num_bins-1] giving the differential entropy (nan for m=0)
      diff_entropy_m0 [1D tf.tensor] of len [num_bins-1] giving the entropy computed for splines with m=0
    """
    # - integral_{x0, x1}(mx+b * log(mx+b)
    # = (mx(mx+2b) - 2*(mx+b)^2*log(mx+b))/4m | {x1,x0}
    # if m=0: - integral_{x0,x1} (b*log(b)) = -(x1-x0)*(b*log(b))
    x_diffs = tf.subtract(x_points[1:], x_points[:-1])
    diff_entropy_m0 = tf.multiply(-x_diffs, tf.multiply(bs[:-1], tf.log(bs[:-1])))
    end_points = tf.where(y_points>0, integral_end_point(x_points, y_points, ms, bs), tf.zeros_like(y_points))
    diff_entropy = tf.subtract(end_points[1:], end_points[:-1])
    return diff_entropy, diff_entropy_m0

def preprocess_points(x_points, y_points):
    """
    Normalize and remove edge cases for histogram points
    Normalizes area to equal 1
    Sets all y_points that are < 0 to 0 and recomputes x_points accordingly
    Inputs:
      x_points [1D tf.tensor] of len [num_bins] containing histogram bin edges
      y_points [1D tf.tensor] of len [num_bins] containing histogram counts
    Outputs:
      tuple containing renormalized (x_points, y_points) 
    """
    ## Normalize
    # area = integral_{x1,x0}(mx+b) = 1/2mx^2+b | {x1, x0}
    # area_new = area_old/area_total
    # area_old = 0.5*m_old*(x1^2-x0^2) + b_old*(x1-x0)
    # b_new = area_old/(area_total*(x1-x0)) - (0.5*m_old*(x1^2-x0^2))/(x1-x0)
    ms, bs = get_line_eq(x_points, y_points)
    x_sq_diffs = tf.subtract(tf.square(x_points[1:]), tf.square(x_points[:-1]))
    x_diffs = tf.subtract(x_points[1:], x_points[:-1])
    areas = tf.add(tf.multiply(0.5, tf.multiply(ms[:-1], x_sq_diffs)), tf.multiply(bs[:-1], x_diffs))
    area_total = tf.reduce_sum(areas)
    left_side = tf.divide(areas, tf.multiply(area_total, x_diffs))
    right_side = tf.divide(tf.multiply(tf.multiply(0.5, ms[:-1]), x_sq_diffs), x_diffs)
    new_bs = tf.subtract(left_side, right_side)
    new_bs = tf.concat([new_bs, tf.expand_dims(new_bs[-1], axis=0)], axis=0)
    y_points = tf.add(tf.multiply(ms, x_points), new_bs)
    ## Prevent negative Ys (weird edge case from normalizing)
    cond = y_points >= 0.0
    y_points = tf.where(cond, y_points, tf.zeros_like(y_points))
    x_points = tf.where(cond, x_points, tf.divide(tf.subtract(y_points, bs), ms))
    return (x_points, y_points)

def unit_entropy(x_points, y_points, eps=1e-12):
    """
    Calculate individual entropy for each pair of points (i.e. each spline segment)
    Inputs:
      x_points [1D tf.tensor] of len [num_bins] containing histogram bin edges
      y_points [1D tf.tensor] of len [num_bins] containing histogram counts
      eps [float] epsilon to avoid log(0) and divide(0) errors
    Outputs:
      unit_entropys [tf.tensor] of shape [num_bins] containing entropy for each spline segment
    """
    x_points, y_points = preprocess_points(x_points, y_points)
    ms, bs = get_line_eq(x_points, y_points)
    entropy, entropy_m0 = get_differential_entropy(x_points, y_points, ms, bs) 
    unit_entropies = tf.where(ms[:-1]==0 and bs[:-1]==0, tf.zeros_like(entropy),
      tf.where(ms[:-1]==0 and bs[:-1]!=0, entropy_m0, entropy))
    return unit_entropies, entropy_m0, entropy

def calc_entropy(u_val, num_bins=50):
    """
    Calculate differential entropy of input vector, u_val
    PDF of histogram of u_val is approximated by a series of splines between histogram points
    Inputs:
      u_val [tf.tensor] of shape [batch_size, num_units]
      num_bins [int] number of bins to use in the histogram
    Outputs:
      entropy [float] differential entropy 
    """
    value_range = [tf.reduce_min(u_val), tf.reduce_max(u_val)]
    hist = tf.histogram_fixed_width(u_val, value_range=value_range, nbins=num_bins)
    bin_edges = tf.linspace(start=value_range[0], stop=value_range[1], num=num_bins)
    hist = tf.to_float(tf.divide(hist, tf.reduce_sum(hist)))
    unit_entropies, entropy_m0, entropy = unit_entropy(bin_edges, hist, eps=1e-12)
    entropy = tf.reduce_sum(unit_entropies)
    #TODO: only return entropy - hist & bins is for debugging
    return (entropy, hist, bin_edges, unit_entropies, entropy_m0, entropy)