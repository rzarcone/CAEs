import tensorflow as tf

def thetas(num_latent, num_tri):
    return tf.Variable(tf.ones((num_latent, num_tri)), name="thetas")

def weights(thetas):
    return tf.exp(thetas)

def zeta(thetas):
    """
    Normalizing constant
    Input:
        thetas [num_latent, num_tri]
    Output:
        zeta [num_latent]
    """
    return tf.reduce_sum(weights(thetas), axis=[1])

def eval_triangle(x, h, n):
    """
    Compute triangle histogram for given latent variables
    Input:
        x [num_batch, num_latent] latent values
        h [num_latent, num_tri] triangle heights
        n [num_tri] number of triangles to use
    x is broadcasted to [num_batch, num_latent, num_tri] (replicated num_tri times)
    h is broadcasted to [num_batch, num_latent, num_tri] (replicated num_batch times)
    n is boradcasted to [num_batch, num_latent, num_tri] (replicated num_batch * num_latent times)
    Output:
        y [num_batch, num_latent, num_tri] evaluated triangles
    """
    h = tf.expand_dims(h, axis=0)
    x = tf.expand_dims(x, axis=-1)
    n = tf.expand_dims(tf.expand_dims(n, axis=0), axis=0)
    y = tf.nn.relu(tf.subtract(h, tf.abs(tf.subtract(x, n,
      name="n_sub"), name="abs_shifted"), name="h_sub"), name="tri_out")
    return y

def prob_est(latent_vals, thetas, tri_locs):
    """
    Inputs:
        latent_vals [num_batch, num_latent] latent values
        thetas [num_latent, num_tri] triangle weights
        tri_locs [num_tri] location of each triangle for latent discretization
    Outputs:
        prob_est [num_batch, num_latent]
    """
    tris = eval_triangle(latent_vals, weights(thetas), tri_locs) # [num_batch, num_latent, num_tri]
    prob_est = tf.divide(tf.reduce_sum(tris, axis=[2]), tf.expand_dims(zeta(thetas), axis=0), name="prob_est")
    return prob_est

def log_likelihood(latent_vals, thetas, tri_locs):
    """
    Inputs:
        latent_vals [num_batch, num_latent] latent values
        thetas [num_latent, num_tri] triangle weights
        tri_locs [num_tri] location of each triangle for latent discretization
    Outputs:
        log_likelihood [num_latent]
    """
    probs = prob_est(latent_vals, weights(thetas), tri_locs) # [num_batch, num_latent]
    return tf.reduce_sum(tf.log(probs, name="log_probs"), axis=[0], name="log_likelihood")

def mle(log_likelihood, thetas, learning_rate):
    grads = tf.gradients(log_likelihood, thetas)[0]
    return thetas.assign_add(tf.multiply(tf.constant(learning_rate), grads))

def calc_entropy(probs):
    """
    Inputs:
        probs [num_batch, num_latent]
    Outputs:
        entropy [num_latent]
    """
    plogp = tf.multiply(probs, tf.log(probs), name="plogp")
    plogp_zeros = tf.where(tf.less_equal(probs, tf.zeros_like(probs)), tf.zeros_like(plogp), plogp, name="plogp_select")
    return -tf.reduce_sum(plogp_zeros, axis=[0], name="entropy")