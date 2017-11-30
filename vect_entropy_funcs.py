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
    ms = tf.concat([ms, tf.expand_dims(ms[-1], axis=0)], axis=0, name="ms")
    bs = tf.subtract(y_points, tf.multiply(ms, x_points), name="bs")
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
    # if m=0: - integral_{x0,x1} (b*log(b)) = -(x1-x0)*(b*log(b))
    m_x = tf.multiply(x_points, ms, name="m_times_x")
    left_side = tf.multiply(m_x, tf.add(m_x, tf.multiply(2.0, bs)), name="integral_left")
    right_side = tf.where(tf.greater(y_points, tf.zeros_like(y_points)), # where y > 0
        tf.multiply(tf.multiply(2.0, tf.square(y_points)), tf.log(y_points)), # true: 2y**2log(y)
        tf.zeros_like(y_points), name="integral_right") # false: 0
    end_points_neq_0 = tf.divide(tf.subtract(left_side, right_side), tf.multiply(4.0, ms), name="integral_end_points")
    end_points_m0 = tf.multiply(-x_points, tf.multiply(bs, tf.log(bs)), name="integral_end_points_m0")
    ms_eq_0 = tf.equal(ms, tf.zeros_like(ms), name="ms_eq_0")
    bs_eq_0 = tf.equal(bs, tf.zeros_like(bs), name="bs_eq_0")
    cond1 = tf.logical_and(ms_eq_0, bs_eq_0, name="ms_and_bs_eq_0")
    cond2 = tf.logical_and(ms_eq_0, tf.logical_not(bs_eq_0, name="bs_neq_0"), name="ms_eq_0_bs_neq_0")
    end_points = tf.where(cond1, tf.zeros_like(x_points),
        tf.where(cond2, end_points_m0, end_points_neq_0, name="end_select_inner"), name="end_select_final")
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
    """
    end_points = tf.identity(integral_end_point(x_points, y_points, ms, bs), name="end_points")
    diff_entropy = tf.subtract(end_points[1:], end_points[:-1], name="diff_entropy")
    return diff_entropy

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

    #NP version
    #fine_xs = np.concatenate([np.arange(x_start, x_end, 1e5, dtype=np.float32) for (x_start, x_end) in zip(x_points[:-1], x_points[1:])])
    #fine_ys = [m*x+b for m in ms for x in fine_xs for b in bs]
    #fine_ys = np.divide(fine_ys, np.sum(fine_ys))
    #new_ys = fine_ys[x_points]

    #TF Version
    #fine_vars = [(np.arange(x_start, x_end, 1e5, dtype=np.float32), ms[idx], bs[idx])
    #    for idx, (x_start, x_end) in enumerate(zip(x_points[:-1], x_points[1:]))]
    #fine_xs = tf.constant(np.concatenate([fine_var[0] for fine_var in fine_vars], dtype=np.float32,
    #  name="fine_xs"))
    #fine_ms = tf.constant(np.concatenate([fine_var[1] for fine_var in fine_vars], dtype=np.float32)
    #fine_bs = tf.constant(np.concatenate([fine_var[2] for fine_var in fine_vars], dtype=np.float32)
    #fine_ys = tf.add(tf.multiply(fine_ms, fine_xs), fine_bs, name="fine_ys")
    #norm_ys = tf.divide(fine_ys, tf.reduce_sum(fine_ys, name="total_area"), name="norm_ys")
    #new_ys = norm_ys[0:-1:1e5]
    
    x_sq_diffs = tf.subtract(tf.square(x_points[1:]), tf.square(x_points[:-1]))
    x_diffs = tf.subtract(x_points[1:], x_points[:-1])
    areas = tf.add(tf.multiply(0.5, tf.multiply(ms[:-1], x_sq_diffs)), tf.multiply(bs[:-1], x_diffs))
    area_total = tf.reduce_sum(areas)
    left_side = tf.divide(areas, tf.multiply(area_total, x_diffs), name="preproc_left_side")
    right_side = tf.divide(tf.multiply(tf.multiply(0.5, ms[:-1]), x_sq_diffs), x_diffs, name="preproc_right_side")
    new_bs = tf.subtract(left_side, right_side)
    new_bs = tf.concat([new_bs, tf.expand_dims(new_bs[-1], axis=0)], axis=0, name="new_bs")
    y_points = tf.add(tf.multiply(ms, x_points), new_bs)
    ## Prevent negative Ys (weird edge case from normalizing)
    cond = tf.greater_equal(y_points, tf.zeros_like(y_points), name="neg_y_cond")
    y_points = tf.where(cond, y_points, tf.zeros_like(y_points), name="processed_y_points")
    return (x_points, y_points)

def unit_entropy(x_points, y_points):
    """
    Calculate individual entropy for each pair of points (i.e. each spline segment)
    Inputs:
      x_points [1D tf.tensor] of len [num_bins] containing histogram bin edges
      y_points [1D tf.tensor] of len [num_bins] containing histogram counts
    Outputs:
      unit_entropy [tf.tensor] of shape [num_bins] containing entropy for each spline segment
    """
    proc_x_points, proc_y_points = preprocess_points(x_points, y_points)
    x_points = tf.identity(proc_x_points, name="proc_x_points")
    y_points = tf.identity(proc_y_points, name="proc_y_points")
    proc_ms, proc_bs = get_line_eq(x_points, y_points)
    ms = tf.identity(proc_ms, name="proc_ms")
    bs = tf.identity(proc_bs, name="proc_bs")
    unit_entropy = get_differential_entropy(x_points, y_points, ms, bs) 
    return unit_entropy

def hist_entropy(x_points, y_points):
    """
    Calculate entropy estimate using normalized histogram heights
    """
    y_points = tf.divide(y_points, tf.reduce_sum(y_points, name="hist_area"), name="normed_heights")
    entropies = tf.where(tf.equal(y_points, tf.zeros_like(y_points)),
        tf.zeros_like(y_points), tf.multiply(-y_points, tf.log(y_points)), name="entropy")
    return entropies

def hist_calc(u_val, num_bins):
    value_range = [tf.reduce_min(u_val), tf.reduce_max(u_val)]
    hist = tf.histogram_fixed_width(u_val, value_range=value_range, nbins=num_bins)
    bin_edges = tf.linspace(start=value_range[0], stop=value_range[1], num=num_bins)
    hist = tf.to_float(tf.divide(hist, tf.reduce_sum(hist)), name="norm_hist")
    return bin_edges, hist
                          
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
    x_points, y_points = hist_calc(u_val, num_bins)
    #unit_entropies = unit_entropy(x_points, y_points)
    unit_entropies = hist_entropy(x_points, y_points)
    entropy = tf.reduce_sum(unit_entropies, name="ent_sum")
    return (entropy, x_points, y_points)
