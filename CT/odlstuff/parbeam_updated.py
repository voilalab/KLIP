import numpy as np
import odl
from operator2 import *
from scipy import interpolate
import torch
import torch.nn.functional as F
import scipy.io

class parbeam(object):
    def __init__(self, sigSize, numAngles, numDetector,
                 min_pt=[-20, -20], max_pt=[20, 20],
                 y_area=[-40, 40], Hf=None, lact=False):
        self.sigSize = sigSize
        self.numAngles = numAngles
        self.numDetector = numDetector
        self.Hf = Hf
        self.name = 'parbeam'
        
        # Define a fixed list of 32 projection angles
        if lact:
            # For lact case, use angles in [0, 2π/3]
            self.angle_list = np.linspace(0, np.pi * 2/3, 32)
        else:
            # For normal case, use angles in [0, π]
            self.angle_list = np.linspace(0, np.pi, 32)
        
        # Select the first numAngles from the predefined list
        if numAngles > 32:
            raise ValueError(f"numAngles ({numAngles}) cannot be greater than 32")
        selected_angles = self.angle_list[:numAngles]
        
        self.y_area = y_area
        self.reco_space = odl.uniform_discr(min_pt=min_pt, max_pt=max_pt, shape=sigSize, dtype='float32')
        
        # Create angle partition from selected angles
        self.angle_partition = odl.nonuniform_partition(selected_angles)
        self.detector_partition = odl.uniform_partition(self.y_area[0], self.y_area[1], self.numDetector)

        # Print debugging information
        print(f"Using {numAngles} angles from predefined list of 32")
        print(f"Selected angles (degrees): {np.degrees(selected_angles)}")

        self.geometry = odl.tomo.Parallel2dGeometry(self.angle_partition, self.detector_partition)
        #self.geometry = odl.tomo.ConeBeamGeometry(self.angle_partition, self.detector_partition,
                                                  #src_radius=133, det_radius=392, axis=axis)

        # A operator
        self.A = odl.tomo.RayTransform(self.reco_space, self.geometry, impl='astra_cuda')

    def grad(self, x, y):
        if self.Hf is None: # AT*(Ax - y)
            Ax = self.fmult(x, self.A)
            grad_d = self.ftran(Ax - y, self.A)
        else: # AT*HT*H*(Ax - y)
            pass
        return grad_d

    def eval(self, x, y):
        if self.Hf is None:
            Ax = self.fmult(x, self.A)
            d = torch.square(Ax - y)
            d = 1 / (2 * d.shape[0]) * torch.sum(d)
        else:
            pass
        return d

    def fmult(self, x, A): # x: B X Y Z
        Ax = OperatorFunction.apply(A, x)
        return Ax

    def ftran(self, z, A):# z: B A H W
        ATx = OperatorFunction.apply(A.adjoint, z)
        return ATx

    def Adagger(self, y, method='fbp', freq=0.9): # y: A H W
        if method == 'fbp':
            fbp_op = odl.tomo.fbp_op(self.A, filter_type='Hann', frequency_scaling=freq)
            x_init = OperatorFunction.apply(fbp_op, y)
        return x_init

    def Atimes(self, x): #return A*x
        return self.fmult(x, self.A)

    def ATtimes(self, y): #return A*x
        return self.ftran(y, self.A)
    
    def get_angle_info(self):
        """Return information about the angles being used"""
        selected_angles = self.angle_list[:self.numAngles]
        return {
            'total_angles_available': len(self.angle_list),
            'num_angles_used': self.numAngles,
            'angles_radians': selected_angles,
            'angles_degrees': np.degrees(selected_angles),
            'angle_spacing_degrees': np.degrees(selected_angles[1] - selected_angles[0]) if len(selected_angles) > 1 else 0
        }
