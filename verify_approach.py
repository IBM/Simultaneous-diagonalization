# Copyright 2022 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is part of the code to reproduce the results in the paper:
# E. van den Berg and Kristan Temme, "Circuit optimization of Hamiltonian
# simulation by simultaneous diagonalization of Pauli clusters," Quantum 4,
# p. 322, 2020. https://doi.org/10.22331/q-2020-09-12-322

from cl import *
import cl_qiskit as clq
import cl_methods as clm
import qiskit
import numpy as np

# Generate test problems (wide, square, tall)
n = 8
for m in [3,8,15] :
   print("# ====================================================")
   print("Problem size %dx%d" % (m,n))
   print("# ====================================================")

   B = generate_test_basis(n,0)
   C = generate_full_rank_weights(m,n,seed=23)
   M = np.dot(C,B) % 2
   T = Tableau(M)
   weights = 0.1 + 0.05 * np.arange(T.m)
   paulis  = matrix_to_pauli(M)
   X = [pauli_to_numpy(p.lower()) for p in paulis]


   # ====================================================
   # Generate baseline solution
   # ====================================================
   N = X[0].shape[0]
   Soln1 = np.eye(N,N,dtype=X[0].dtype)
   for (x,gamma) in zip(X,weights) :
      # Compute eigendecomposition
      (Ev,Eb) = np.linalg.eig(x)
      # We have x = Eb * diag(Ev) * Eb':
      xHat = np.dot(Eb,np.dot(np.diag(Ev),Eb.T.conj()))
      print("Sanity check ||x-xHat|| = %s" % str(np.linalg.norm(xHat - x,'fro')))
      expx = np.dot(Eb,np.dot(np.diag(np.exp(1j*gamma*Ev)), Eb.T.conj()))
      Soln1 = np.dot(expx, Soln1)


   # ====================================================
   # Generate circuit for direct exponentiation
   # ====================================================
   for optimized in [False, True] :
      R = clq.RecordCircuit(T.n+1)
      clm.circuitDirectExponentiation(R, paulis, weights, optimized=optimized)

      backend = qiskit.Aer.get_backend('unitary_simulator')
      result = qiskit.execute(R.circuit, backend).result()
      Soln2 = result.get_unitary()
      Soln2 = Soln2[:N,:N]
      print("Difference ||Soln2 - Soln1|| = %s" % str(np.linalg.norm(Soln2-Soln1,'fro')))


   # ====================================================
   # Approach based on simulatenous diagonalization
   # ====================================================
   for method in ['gmc','greedy1','greedy2','cnot','cnot_log2','cnot_best'] :
      if ((method == 'gmc') and (m != n)) :
         continue

      for funOrderZ in [None, orderZ] :
         R = clq.RecordCircuit(T.n+1)
         clm.circuitDiagExponentiation(R, paulis, weights, method=method, funOrderZ=funOrderZ)

         backend = qiskit.Aer.get_backend('unitary_simulator')
         result = qiskit.execute(R.circuit, backend).result()
         Soln3 = result.get_unitary()
         Soln3 = Soln3[:N,:N]
         print("Difference ||Soln3 - Soln1|| = %s" % str(np.linalg.norm(Soln3-Soln1,'fro')))
