import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np 
import vect_entropy_funcs as vf
import tensorflow as tf

class EntTest(tf.test.TestCase):
    def get_xs_ys_ms_bs(self):
        hand_xs = np.arange(5, dtype=np.float32)
        hand_ys = np.array([0,3,0,0,5], dtype=np.float32)
        """
        y = mx + b 
        b = y - mx
        m = (y - b)/x = (y_{n} - y_{n-1})/(x_{n} - x_{n-1})
        """
        hand_ms = np.array([3,-3,0,5,5], dtype=np.float32)
        hand_bs = np.array([0,6,0,-15,-15], dtype=np.float32)

        ### Random version ###
        rand_xs = np.arange(1000, dtype=np.float32)
        rand_ys = np.random.normal(0, 10, 1000).astype(np.float32)
        rand_ms = np.divide(rand_ys[1:]-rand_ys[:-1], rand_xs[1:]-rand_xs[:-1], dtype=np.float32)
        rand_ms = np.append(rand_ms, rand_ms[-1])
        rand_bs = rand_ys-rand_ms*rand_xs

        return [[hand_xs, hand_ys, hand_ms, hand_bs], [rand_xs, rand_ys, rand_ms, rand_bs]]

    def test_get_line_eq(self):
        num_runs = 1
        for run in range(num_runs):
            print("Line test ", run, " out of ", num_runs)
            inputs = self.get_xs_ys_ms_bs()
            for x_points, y_points, ms_gt, bs_gt in inputs:
                with self.test_session() as sess:
                    xs_var = tf.Variable(x_points)
                    ys_var = tf.Variable(y_points)
                    ms_gt_var = tf.Variable(ms_gt)
                    bs_gt_var = tf.Variable(bs_gt)
                    ms, bs = vf.get_line_eq(xs_var, ys_var)
                    tf.global_variables_initializer().run()
                    self.assertAllEqual(ms_gt, ms.eval())
                    self.assertAllEqual(bs_gt, bs.eval())

    def test_integral_end_point(self):
        """
        end_point = (mx(mx+2b)-2y^2log(y))/(4m)
        """
        old_err = np.seterr(all='ignore')
        num_runs = 1
        for run in range(num_runs):
            print("End point test ", run, " out of ", num_runs)
            inputs = self.get_xs_ys_ms_bs()
            hand_xs, hand_ys, hand_ms, hand_bs = inputs[0]
            end_0 = np.divide(hand_ms[0]*hand_xs[0]*(hand_ms[0]*hand_xs[0]+2*hand_bs[0])-2*0, 4*hand_ms[0], dtype=np.float32)
            end_1 = np.divide(hand_ms[1]*hand_xs[1]*(hand_ms[1]*hand_xs[1]+2*hand_bs[1])-2*hand_ys[1]**2*np.log(hand_ys[1]), 4*hand_ms[1], dtype=np.float32)
            end_2 = np.divide(hand_ms[2]*hand_xs[2]*(hand_ms[2]*hand_xs[2]+2*hand_bs[2])-2*0, 4*hand_ms[2], dtype=np.float32)
            end_3 = np.divide(hand_ms[3]*hand_xs[3]*(hand_ms[3]*hand_xs[3]+2*hand_bs[3])-2*0, 4*hand_ms[3], dtype=np.float32)
            end_4 = np.divide(hand_ms[4]*hand_xs[4]*(hand_ms[4]*hand_xs[4]+2*hand_bs[4])-2*hand_ys[4]**2*np.log(hand_ys[4]), 4*hand_ms[4], dtype=np.float32)
            ## Random version
            rand_xs, rand_ys, rand_ms, rand_bs = inputs[1]
            ylogy = np.where(rand_ys<=0, np.zeros_like(rand_ys, dtype=np.float32), np.multiply(rand_ys**2, np.log(rand_ys), dtype=np.float32))
            rand_ep = np.divide(rand_ms*rand_xs*(rand_ms*rand_xs+2*rand_bs)-2*ylogy, 4*rand_ms, dtype=np.float32)
            ## Add GT to inputs
            inputs[0] += [np.array([end_0,end_1,end_2,end_3,end_4], dtype=np.float32)]
            inputs[1] += [rand_ep]
            ## Run tests 
            for x_points, y_points, ms_gt, bs_gt, ep_gt in inputs:
                with self.test_session() as sess:
                    xs_var = tf.Variable(x_points)
                    ys_var = tf.Variable(y_points)
                    ms_gt_var = tf.Variable(ms_gt)
                    bs_gt_var = tf.Variable(bs_gt)
                    ep_gt_var = tf.Variable(ep_gt)
                    ep = vf.integral_end_point(xs_var, ys_var, ms_gt_var, bs_gt_var)
                    # TODO: Might be better to modify _assertArrayLikeAllClose() to include an "allowNAN" flag?
                    atol=1e-6
                    rtol=1e-6
                    close_eps_check = tf.less(tf.abs(ep - ep_gt_var), atol+rtol*tf.abs(ep_gt_var))
                    test_eps = tf.equal(tf.logical_or(close_eps_check,
                        tf.logical_and(tf.is_nan(ep), tf.is_nan(ep_gt_var))), tf.Variable(True))
                    tf.global_variables_initializer().run()
                    self.assertAllEqual([True,]*len(x_points), test_eps.eval())
        _ = np.seterr(**old_err)

    def test_preprocess_points(self):
        num_runs = 50
        for run in range(num_runs):
            print("End point test ", run, " out of ", num_runs)
            inputs = self.get_xs_ys_ms_bs()
            hand_xs, hand_ys, hand_ms, hand_bs = inputs[0]
            a0 = (0.5*hand_ms[0]*hand_xs[1]**2+hand_bs[0]*hand_xs[1])-(0.5*hand_ms[0]*hand_xs[0]**2+hand_bs[0]*hand_xs[0])
            a1 = (0.5*hand_ms[1]*hand_xs[2]**2+hand_bs[1]*hand_xs[2])-(0.5*hand_ms[1]*hand_xs[1]**2+hand_bs[1]*hand_xs[1])
            a2 = (0.5*hand_ms[2]*hand_xs[3]**2+hand_bs[2]*hand_xs[3])-(0.5*hand_ms[2]*hand_xs[2]**2+hand_bs[2]*hand_xs[2])
            a3 = (0.5*hand_ms[3]*hand_xs[4]**2+hand_bs[3]*hand_xs[4])-(0.5*hand_ms[3]*hand_xs[3]**2+hand_bs[3]*hand_xs[3])
            a_vals = np.array([a0,a1,a2,a3],dtype=np.float32)  
            b0 = (a_vals[0])/((hand_xs[1]-hand_xs[0])*np.sum(a_vals))-(hand_ms[0]*(hand_xs[1]**2-hand_xs[0]**2))/(2*(hand_xs[1]-hand_xs[0]))
            b1 = (a_vals[1])/((hand_xs[2]-hand_xs[1])*np.sum(a_vals))-(hand_ms[1]*(hand_xs[2]**2-hand_xs[1]**2))/(2*(hand_xs[2]-hand_xs[1]))
            b2 = (a_vals[2])/((hand_xs[3]-hand_xs[2])*np.sum(a_vals))-(hand_ms[2]*(hand_xs[3]**2-hand_xs[2]**2))/(2*(hand_xs[3]-hand_xs[2]))
            b3 = (a_vals[3])/((hand_xs[4]-hand_xs[3])*np.sum(a_vals))-(hand_ms[3]*(hand_xs[4]**2-hand_xs[3]**2))/(2*(hand_xs[4]-hand_xs[3]))
            b_vals = np.array([b0,b1,b2,b3,b3],dtype=np.float32)
            hand_ys_gt = hand_ms*hand_xs+b_vals
            hand_ys_gt = np.where(hand_ys_gt<0, np.zeros_like(hand_ys_gt, dtype=np.float32), hand_ys_gt.astype(np.float32))
            ## Random version
            rand_xs, rand_ys, rand_ms, rand_bs = inputs[1]
            
            
            
            #rand_as = (0.5*rand_ms[:-1]*rand_xs[1:]**2+rand_bs[:-1]*rand_xs[1:])-(0.5*rand_ms[:-1]*rand_xs[:-1]**2+rand_bs[:-1]*rand_xs[:-1])

            #rand_as2 = 0.5*rand_ms[:-1]*rand_xs[1:]**2+rand_bs[:-1]*rand_xs[1:]-0.5*rand_ms[:-1]*rand_xs[:-1]**2-rand_bs[:-1]*rand_xs[:-1]
   
            
            rand_as = 0.5*rand_ms[:-1]*(rand_xs[1:]**2-rand_xs[:-1]**2)+rand_bs[:-1]*(rand_xs[1:]-rand_xs[:-1])
            
            
            #assert (rand_as2==rand_as).all(), str(rand_as-rand_as2)
            
            
            
            rand_new_bs = (rand_as)/((rand_xs[1:]-rand_xs[:-1])*np.sum(rand_as))-(rand_ms[:-1]*(rand_xs[1:]**2-rand_xs[:-1]**2))/(2*(rand_xs[1:]-rand_xs[:-1]))
            rand_new_bs = np.append(rand_new_bs,rand_new_bs[-1])
            rand_ys_gt = rand_ms*rand_xs+rand_new_bs
            rand_ys_gt = np.where(rand_ys_gt<0, np.zeros_like(rand_ys_gt, dtype=np.float32), rand_ys_gt.astype(np.float32))
            ## Add GT to inputs
            inputs[0] += [hand_ys_gt]
            inputs[1] += [rand_ys_gt]
            ## Run tests 
            for x_points, y_points, ms_gt, bs_gt, ys_gt in inputs:
                with self.test_session() as sess:
                    xs_var = tf.Variable(x_points)
                    ys_var = tf.Variable(y_points)
                    xs, ys = vf.preprocess_points(xs_var, ys_var)
                    tf.global_variables_initializer().run()
                    self.assertAllClose(ys_gt,ys.eval(), rtol=1e-4)
            
       

            
            
            
if __name__ == "__main__":
    tf.test.main()