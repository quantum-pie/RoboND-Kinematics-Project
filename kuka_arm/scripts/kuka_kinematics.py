#!/usr/bin/env python

# import modules
import numpy as np
from sympy import symbols, cos, sin, pi, sqrt, atan2, simplify
from sympy.matrices import Matrix
from sympy.solvers import solve
import tf
import argparse
import rospy

def spherical_wrist_inverse(R):
    print("R: " + str(R))
    c5 = R[1,2]
    s5 = +np.sqrt(1 - c5 ** 2)
    theta_5 = np.arctan2(s5, c5)

    r22 = -np.sign(s5) * R[1,1]
    r21 = np.sign(s5) * R[1,0]
    r33 = np.sign(s5) * R[2,2]
    r13 = -np.sign(s5) * R[0,2]

    theta_4 = np.arctan2(+R[2,2], -R[0,2])
    theta_6 = np.arctan2(-R[1,1], +R[1,0])
    return [theta_4, theta_5, theta_6]


class KukaKinematics:
    def __init__(self):
        ### Create symbols for joint variables
        self.theta = symbols('theta1:8')
        self.d = symbols('d1:8')
        self.a = symbols('a0:7')
        self.alpha = symbols('alpha0:7')

	# DH Parameters
        pi_2 = 0.5 * np.pi
	self.s = {self.alpha[0]: 0,      self.a[0]:   0,    self.d[0]: 0.75, 
	     self.alpha[1]: -pi_2,  self.a[1]: 0.35,   self.d[1]: 0,     self.theta[1]: self.theta[1] - pi_2,  
	     self.alpha[2]: 0,      self.a[2]: 1.25,   self.d[2]: 0,
	     self.alpha[3]: -pi_2,  self.a[3]: -0.054, self.d[3]: 1.5, 
             self.alpha[4]: pi_2,   self.a[4]: 0,      self.d[4]: 0,
             self.alpha[5]: -pi_2,  self.a[5]: 0,      self.d[5]: 0,
             self.alpha[6]: 0,      self.a[6]: 0,      self.d[6]: 0.303, self.theta[6]: 0}

        T = list()
        for alpha_i1, a_i1, d_i, theta_i in zip(self.alpha, self.a, self.d, self.theta):
            theta_sin = sin(theta_i)
            theta_cos = cos(theta_i)
            alpha_sin = sin(alpha_i1)
            alpha_cos = cos(alpha_i1) 
            T_i1_i = Matrix([[ theta_cos, -theta_sin,            0,           a_i1],
                [ theta_sin*alpha_cos, theta_cos*alpha_cos, -alpha_sin, -alpha_sin*d_i],
                [ theta_sin*alpha_sin, theta_cos*alpha_sin,  alpha_cos,  alpha_cos*d_i],
                [                   0,                   0,          0,              1]])
            T.append(T_i1_i.subs(self.s))

        self.T_0_3 = T[0] * T[1] * T[2]
        self.T_0_6 = self.T_0_3 * T[3] * T[4] * T[5]
        self.T_0_7 = self.T_0_6 * T[6]
        #print(self.T_3_6)

        self.T_dh_gz = np.array([[0, 0, 1, 0],
            [0, -1, 0, 0],
            [1,  0, 0, 0],
            [0,  0, 0, 1]])

    def forward(self, theta_val):
        thetas = {self.theta[i]: float(theta_val[i]) for i in range(6)}
        T_0_7_eval = np.array(self.T_0_7.evalf(subs=thetas), dtype=float)
        T_0_gz = T_0_7_eval.dot(self.T_dh_gz)
        effector_position = T_0_gz[0:3, 3]
        effector_orientation = tf.transformations.quaternion_from_matrix(T_0_gz)

        return effector_position, effector_orientation

    def forward_wc(self, theta_val):
        thetas = {self.theta[i]: float(theta_val[i]) for i in range(6)}
        T_0_6_eval = np.array(self.T_0_6.evalf(subs=thetas), dtype=float)
        wc_position = T_0_6_eval[0:3, 3]
        wc_orientation = tf.transformations.quaternion_from_matrix(T_0_6_eval)

        return wc_position, wc_orientation

    def inverse(self, target_position, target_orientation):
        target_R = tf.transformations.quaternion_matrix(target_orientation)[0:3, 0:3]
        R_0_6 = np.dot(target_R, np.transpose(self.T_dh_gz[0:3, 0:3]))
        wc_global = target_position - np.dot(R_0_6, np.array([0, 0, self.s[self.d[6]]]))

        theta_1 = np.arctan2(wc_global[1], wc_global[0])

        # in global frame if theta_1 was 0
        wc_x = np.linalg.norm(wc_global[0:2])
        wc_y = wc_global[2]

        # move origin to joint 2
        wc_x -= self.s[self.a[1]]
        wc_y -= self.s[self.d[0]]

        wc_len = np.sqrt(wc_x ** 2 + wc_y ** 2)
        wc_angle = np.arctan2(wc_y, wc_x)

        link_len_3wc = np.sqrt(self.s[self.a[3]] ** 2 + self.s[self.d[3]] ** 2)

        link_len_23 = self.s[self.a[2]]

        # angle opposite 3-wc link
        gamma = np.arccos(0.5 * (link_len_23 ** 2 + wc_len ** 2 - link_len_3wc ** 2) / (link_len_23 * wc_len))

        theta_2 = 0.5 * np.pi - (gamma + wc_angle)

        # angle opposite wc-o link
        phi = np.arccos(0.5 * (link_len_23 ** 2 + link_len_3wc ** 2 - wc_len ** 2) / (link_len_23 * link_len_3wc))
        #phi = np.arcsin(wc_len * np.sin(gamma) / link_len_3wc)
        psi = np.arctan(-self.s[self.a[3]] / self.s[self.d[3]])
        theta_3 = 0.5 * np.pi - (phi + psi)

        thetas = {self.theta[0]: theta_1, self.theta[1]: theta_2, self.theta[2]: theta_3}
        R_3_0 = np.transpose(np.array(self.T_0_3.evalf(subs=thetas), dtype=float)[0:3, 0:3])

        R_3_6 = np.dot(R_3_0, R_0_6)

        final_angles = spherical_wrist_inverse(R_3_6)
        result = [theta_1, theta_2, theta_3, final_angles[0], final_angles[1], final_angles[2]]
        rospy.loginfo("Thetas: " + str(result))
        return result

if __name__ == "__main__":
    kinematics = KukaKinematics()
    parser = argparse.ArgumentParser(
        description='''This script performs forward kinematics of Kuka robot''')
    parser.add_argument('joint_angles', nargs='+',
                        help='Set of joint angles (radians) in ascending joint order (from base to effector)')

    clargs = parser.parse_args()
    if len(clargs.joint_angles) != 6:
        raise argparse.ArgumentTypeError('Wrong number of joint angles!')

    trans, orient = kinematics.forward(clargs.joint_angles)
    print("Effector translation: " + str(trans))
    print("Effector orientation: " + str(orient))

    new_thetas = kinematics.inverse(trans, orient)
    print("New thetas: " + str(new_thetas))

    new_trans, new_orient = kinematics.forward(new_thetas)
    print("New effector translation: " + str(new_trans))
    print("New effector orientation: " + str(new_orient))
    


    
