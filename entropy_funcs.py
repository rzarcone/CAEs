import tensorflow as tf
import time as ti

def get_line_eq(x_points, y_points):
    t0 = ti.time()
    m = tf.divide(tf.subtract(y_points[1],y_points[0]), tf.subtract(x_points[1],x_points[0]))
    b = tf.subtract(y_points[1], tf.multiply(m, x_points[1]))
    t1 = ti.time()
    #print("get_line_eq: ",(t1-t0))
    return (m,b)

def get_ms_and_bs(data):
    t0 = ti.time()
    ms = []
    bs = []
    x0s = []
    x1s = []
    for xy_points in data:
        # xy_points[0] are the x values (x0, x1)
        # xy_points[1] are the y values (y0, y1)
        m,b = get_line_eq(xy_points[0], xy_points[1])
        ms.append(m)
        bs.append(b)
        x0s.append(xy_points[0][0])
        x1s.append(xy_points[0][1])
    t1 = ti.time()
    #print("get_ms_and_bs: ",(t1-t0))
    return (ms, bs, x0s, x1s)

def integral_end_point(x,m,b):
    # - integral_{x0,x1}(mx+b * log(mx+b))
    #  = -(2*(mx+b)^2*log(mx+b) - mx(mx+2b))/4m | {x1,x0}
    t0 = ti.time()
    m_x = tf.multiply(m,x)
    y = tf.add(m_x,b)
    two_y_sq = tf.multiply(2.0, tf.square(y))
    left_side = tf.multiply(two_y_sq, tf.log(y))
    two_b = tf.multiply(2.0, b)
    right_side = tf.multiply(m_x, tf.add(m_x, two_b))
    numerator = tf.multiply(-1.0, tf.subtract(left_side, right_side))
    denom = tf.multiply(4.0, m)
    quotient = tf.divide(numerator, denom)
    t1 = ti.time()
    #print("integral_end_point: ",(t1-t0)) 
    return quotient

def get_differential_entropy(x_points, y_points, m, b):
    # - integral_{x0, x1}(mx+b * log(mx+b))
    def m_equal_0():
        def b_equal_0():
            return tf.constant(0, dtype=tf.float32)
        def b_nequal_0():
            # - integral_{x0,x1} (b*log(b)) = -(x1-x0)*(b*log(b))
            x_diff = tf.subtract(x_points[1], x_points[0])
            return tf.multiply(-x_diff, tf.multiply(b, tf.log(b)))
        return tf.cond(tf.equal(b, 0.0), b_equal_0, b_nequal_0)
    def m_nequal_0():
        def y_neg():
            return tf.constant(0, dtype=tf.float32)
        def y_pos():
            return tf.subtract(integral_end_point(x_points[1],m,b),integral_end_point(x_points[0],m,b))
        return tf.cond(tf.logical_or(tf.less_equal(y_points[0],0.0),tf.less_equal(y_points[1],0.0)),y_neg,y_pos)
    t0 = ti.time()
    diff_entropy = tf.cond(tf.equal(m, 0.0), m_equal_0, m_nequal_0)
    t1 = ti.time()
    #print("get_differential_entropy: ", (t1-t0))
    return diff_entropy

def compute_area(ms, bs, x0s, x1s):
    t0 = ti.time()
    total_area=tf.constant(0.0)
    for m,b,x0,x1 in zip(ms,bs,x0s,x1s):
        # area = integral(mx+b) evaluated at x1, x0
        # integral_{x0, x1}(mx+b) = m/2*x^2+bx | {x0, x1}
        # integral = (0.5*m*x1^2+b*x1) - (0.5*m*x0^2+b*x0)
        # integral = (0.5*m*(x1^2 - x0^2)) + (b * (x1-x0))
        left_side = tf.multiply(tf.constant(0.5),tf.multiply(m,tf.subtract(tf.square(x1),tf.square(x0))))
        rigt_side = tf.multiply(b,tf.subtract(x1,x0))        
        unit_area = tf.add(left_side,rigt_side)
        total_area = tf.add(total_area,unit_area)
    t1 = ti.time()
    #print("compute_area: ",(t1-t0))
    return total_area

def get_normalized_points(xy_points, area_total, m0, b0):
    t0 = ti.time()
    # area_new = area_old/area_total
    # area_old = 0.5*m_old*(x1^2-x0^2) + b_old*(x1-x0)
    # b_new = area_old/(area_total*(x1-x0)) - (0.5*m_old*(x1^2-x0^2))/(x1-x0)
    x0 = xy_points[0][0]  
    x1 = xy_points[0][1]
    x_diff = tf.subtract(x1,x0)
    x_sq_diff = tf.subtract(tf.square(x1),tf.square(x0))
    area_old = compute_area([m0],[b0],[x0],[x1])
    left_side = tf.divide(area_old, tf.multiply(area_total, x_diff))
    right_side = tf.divide(tf.multiply(tf.multiply(0.5, m0), x_sq_diff), x_diff)
    new_b = tf.subtract(left_side,right_side)
    new_y0 = tf.add(tf.multiply(m0,xy_points[0][0]),new_b)
    new_y1 = tf.add(tf.multiply(m0,xy_points[0][1]),new_b)
    new_points = [[xy_points[0][0], xy_points[0][1]], [new_y0, new_y1]]
    t1 = ti.time()
    #print("get_normalized_points: ",(t1-t0))
    return new_points

def unit_entropy(data, eps=0):
    t0 = ti.time()
    def cond_diff_entropy(ys, m, b):
        def points_zero():
            return tf.constant(0, dtype=tf.float32)
        def points_nzero():    
            return get_differential_entropy(xy_points[0], tf.add(xy_points[1], eps), m, b)
        zero = tf.constant(0, dtype=tf.float32)
        return tf.cond(tf.equal(zero, tf.reduce_sum(ys)), points_zero, points_nzero)
        # Should replace above code:
        #return tf.cond(tf.equal(0, tf.reduce_sum(ys)), lambda: tf.constant(0, dtype=tf.float32),
        #    get_differential_entropy(xy_points[0], tf.add(xy_points[1], eps), m, b))
    ms, bs, x0s, x1s = get_ms_and_bs(data)
    area = compute_area(ms, bs, x0s, x1s)
    new_data = []
    for idx, xy_points in enumerate(data):
        m0 = ms[idx]
        b0 = bs[idx]
        xy_points = get_normalized_points(xy_points, area, m0, b0)
        new_data.append(xy_points)
    ms, bs, x0s, x1s = get_ms_and_bs(new_data)
    entropy = tf.constant(0.0)
    for idx, xy_points in enumerate(new_data):
        m = ms[idx]
        b = bs[idx]
        entropy = tf.add(entropy, cond_diff_entropy(xy_points[1], m, b))
    t1 = ti.time()
    #print("unit_entropy: ",(t1-t0))
    return entropy

def var(xs,ys):
    t0 = ti.time()
    mean_xs = tf.divide(tf.matmul(xs,tf.transpose(ys)),tf.size(xs))
    norm_mean = tf.square(tf.subtract(xs,mean_xs))
    var_xs = tf.matmul(norm_mean,tf.transpose(ys))
    t1 = ti.time()
    #print("var: "(t1-t0))
    return var_xs

def calc_entropy(u_val, num_bins):
    t0 = ti.time()
    value_range = [tf.reduce_min(u_val), tf.reduce_max(u_val)]
    hist = tf.histogram_fixed_width(u_val, value_range=value_range, nbins=num_bins)
    bin_edges = tf.linspace(start=value_range[0], stop=value_range[1], num=num_bins)
    hist = tf.to_float(tf.divide(hist, tf.reduce_sum(hist)))
    hist_data = []
    for index in range(num_bins-1):
        x_points = [bin_edges[index], bin_edges[index+1]]
        y_points = [hist[index], hist[index+1]]
        hist_data.append([x_points, y_points])
    entropy = unit_entropy(hist_data, eps=1e-12)
    t1 = ti.time()
    #print("calc_entropy", (t1-t0))
    return (entropy, hist, bin_edges)