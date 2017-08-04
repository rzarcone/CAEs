import tensorflow as tf

def get_line_eq(x_points, y_points):
    ms = tf.divide(tf.subtract(y_points[1:], y_points[:-1]), tf.subtract(x_points[1:], x_points[:-1]))
    ms = tf.concat([ms, tf.divide(tf.subtract(y_points[-1], y_points[-2]), tf.subtract(x_points[-1], x_points[-2]))], axis=0)
    bs = tf.subtract(y_points[:-1], tf.multiply(ms, x_points[:-1]))
    bs = tf.concat([bs, tf.subtract(y_points[-1], tf.multiply(ms[-1], x_points[-1]))], axis=0)
    return (ms, bs)

def integral_end_point(x,m,b):
    return quotient

def get_differential_entropy(x_points, y_points, m, b):
    ### TODO
    return diff_entropy, diff_entropy_m0

def normalize_points(x_points, y_points):
    # area = integral_{x1,x0}(mx+b) = 1/2mx^2+b | {x1, x0}
    # area_new = area_old/area_total
    # area_old = 0.5*m_old*(x1^2-x0^2) + b_old*(x1-x0)
    # b_new = area_old/(area_total*(x1-x0)) - (0.5*m_old*(x1^2-x0^2))/(x1-x0)
    ms, bs = get_line_eq(x_points, y_points)
    x_sq_diffs = tf.subtract(tf.square(x_points[1:]), tf.square(x_points[:-1]))
    x_diffs = tf.subtract(x_points[1:], x_points[:-1])
    areas = tf.add(tf.multiply(0.5, tf.multiply(ms, x_sq_diffs)), tf.multiply(bs, x_diffs))
    total_area = tf.reduce_sum(areas)
    left_side = tf.divide(areas, tf.multiply(area_total, x_diffs))
    right_side = tf.divide(tf.multiply(tf.multiply(0.5, ms), x_sq_diffs), x_diffs)
    new_bs = tf.subtract(left_side, right_side)
    new_y_points = tf.add(tf.multiply(ms, x_points), new_bs)
    return (x_points, new_y_points)

def var(xs,ys):
    return var_xs

def unit_entropy(x_points, y_points, eps=0):
    x_points, y_points = normalize_points(x_points, y_points)
    ms, bs = get_line_eq(x_points, y_points)
    entropy, entropy_m0 = get_differential_entropy(x_points, y_points, ms, bs) 
    
    return entropy

def calc_entropy(u_val, num_bins):
    value_range = [tf.reduce_min(u_val), tf.reduce_max(u_val)]
    hist = tf.histogram_fixed_width(u_val, value_range=value_range, nbins=num_bins)
    bin_edges = tf.linspace(start=value_range[0], stop=value_range[1], num=num_bins)
    hist = tf.to_float(tf.divide(hist, tf.reduce_sum(hist)))
    entropy = unit_entropy(bin_edges, hist, eps=1e-12)
    return (entropy, hist, bin_edges)