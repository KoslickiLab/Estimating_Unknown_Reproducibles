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
		if run_now:
			self.run()

	def run(self):
		self.x_opt = self.get_optim()

	def get_optim(self):
		(M,N) = np.shape(self.A)
		x = cp.Variable(N)
		objective = cp.Minimize(cp.norm(cp.atoms.elementwise.maximum.maximum(self.b - (self.A @ x), 0), 1) \
		    + self.w*cp.norm(cp.atoms.elementwise.maximum.maximum((self.A @ x) - self.b, 0), 1))
		constraints = [x >= 0]
		prob = cp.Problem(objective, constraints)
		result = prob.solve(solver = cp.GUROBI)
		prod = self.A@x.value
		return x.value