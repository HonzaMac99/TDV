import tools as tb
import numpy as np

E = np.array([[ 0.05913779, -0.41038986,  0.4183559 ],
              [ 0.34493564,  0.29659218, -0.73345105],
              [-0.7325935,   0.60005267,  0.10680269]])

u1 = np.array([[1.44105122e+00, 1.69310946e-01, 1.89764200e-03, 1.07704372e+00, 1.46356042e+00],
               [6.76344924e-01, 2.34012704e-01, 1.73928972e+00, 9.57981454e-01, 1.97479254e+00],
               [5.49250632e+00, 5.29449686e+00, 5.50936931e+00, 5.04143361e+00, 5.81373599e+00]])

u2 = np.array([[3.17929982, 1.81941743, 2.12525133, 2.84895834, 3.62118962],
               [1.90879731, 1.85713898, 3.34615888, 2.18182892, 3.17116   ],
               [5.31391195, 5.32940958, 5.15811555, 4.83264754, 5.28501216]])

e = tb.EutoRt(E**2, u1, u2)
print(e)
