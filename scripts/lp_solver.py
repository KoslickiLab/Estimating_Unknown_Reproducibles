import numpy as np
import h5py
import gurobipy as gp
from gurobipy import GRB
import cvxpy as cp

#inputs: matrix A, vector b, weight w
#output: estimate vector x and metadata
class lp_solver():
	def __init__(self, A, b, w, run_now = False):
		self.A = A
		self.b = b
		self.w = w
		self.K, self.N = np.shape(self.A)
		if run_now:
			self.x_opt = self.get_optim()


	def get_optim(self):
		x = cp.Variable(self.N)
		u = cp.Variable(self.K)
		v = cp.Variable(self.K)
		tau = 1/(self.w+1)
		ones_K = np.ones(self.K)
		objective = cp.Minimize(
			tau*(ones_K @ u) + (1-tau)*(ones_K @ v)
		)
		constraints = [
			x >= 0,
			u >= 0,
			v >= 0,
			u - v + (self.A @ x) == self.b,
		]
		prob = cp.Problem(objective, constraints)
		result = prob.solve(solver = cp.SCIPY, verbose=False)
		prod = self.A@x.value
		return x.value
