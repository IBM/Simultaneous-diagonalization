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

import cl
import cl_qiskit as clq
import math
import numpy as np


def draw_R(R) :
   C = clq.RecordCircuit(R.nQubits)
   C.reset()
   R.sendTo(C)
   print(C.circuit.draw())


# ======================================================================
# Application of the different methods on a matrix
# ======================================================================

def circuitDiagonalize(T,R,method) :
   T0 = T
   T  = cl.Tableau(T0)
   T.addRecorder(R)

   # Apply the method and check correctness
   method(T)
   T.assertXEmpty()

   # Simplify the circuit and check correctness
   R.simplify()
   R.apply(T0)
   T0.assertXEmpty()


def circuitDiagonalize_CNot(T,R,blocksizes) :
   #R = cl.RecordOperations(T.n) # or T.n + 1
   T0 = T
   T  = cl.Tableau(T0)
   T.addRecorder(R)

   # Sweep the tableau and check correctness
   cl.zeroX_algorithm1_cnot(T)
   T.assertXEmpty()

   # Apply CNot optimization if needed
   optimal_blocksize = []
   if (len(blocksizes) > 0) :
      # Get the stages
      stages = R.getStages()

      #if ((T.n <= T.m) and (len(stages) > 3)) :
      #   raise Exception("The number of stages for circuitDiagonalize_CNot should not exceed three")

      # Simplify the cnot stages
      new_stages = []; cnot_count = 0
      for (idx,stage) in enumerate(stages) :
         if (idx % 2 == 1) :
            cnots = [(op.index1,op.index2) for op in stage]
            best_cnots = cnots
            best_count = len(stage) # Current cnot count
            best_size = 0

            for blocksize in blocksizes :
               cnots_new = cl.reduce_cnot(cnots, T.n, blocksize)
               cl.ensure_cnot_equivalence(cnots,cnots_new,T.n)

               # Keep the best setting
               if (len(cnots_new) < best_count) :
                  best_cnots = cnots_new
                  best_count = len(cnots_new)
                  best_size = blocksize

            # Update the stages using the best cnots
            stage = [cl.OperatorCNot(cnot[0],cnot[1]) for cnot in best_cnots]
            optimal_blocksize.append(best_size)

         # Append the stage
         new_stages.append(stage)

      # Update the circuit
      R.operations = []
      for stage in new_stages :
         R.operations += stage

   # Simplify the circuit and check correctness
   R.simplify()
   R.apply(T0)
   T0.assertXEmpty()

   return optimal_blocksize


def _circuitDiagExponentiateGenerateZ(R, Z, weights) :
   (m,n) = Z.shape
   s = np.zeros(n,dtype=Z.dtype)
   for i in range(m+1) :
      sNew = Z[i,:] if (i < m) else np.zeros(n,dtype=Z.dtype)

      # Finalize previous layer cnots
      idx = list(np.where(s != sNew)[0])
      for j in idx :
         R.addCNot(j,n)

      # Add exponentiation
      if (i < m) :
         R.addRz(n, weights[i])

      # Update the state
      s = sNew


def circuitDiagExponentiateGenerate(R, RDiag, T, weights, funOrderZ=None) :
   # T is a tableau that has already been diagonalized
   # RD are the operations used to diagonalize the operation
   T.assertXEmpty()

   # Get the information
   w = (1 - 2 * T.getSign()) * weights
   Z = T.getZ()
   m = T.m
   n = T.n

   # Reorder the operators if needed
   if (funOrderZ is not None) :
      index = funOrderZ(Z)

      # Validate the ordering
      if ((len(index) != T.m) or (len(set(index)) != T.m)) :
         raise Exception("Invalid reordering of rows in block Z")

      Z = Z[index,:]
      w = w[index]

   # Create a new circuit
   RDiag.sendTo(R)
   _circuitDiagExponentiateGenerateZ(R, Z, w)
   RDiag.inverse().sendTo(R)


def circuitDiagExponentiation(R, paulis, weights, method, funOrderZ=None) :
   T  = cl.Tableau(cl.pauli_to_matrix(paulis))
   RD = cl.RecordOperations(T.n)

   # Generate the circuit for simultaneous diagonalization
   method = method.lower()
   if (method == 'gmc') :
      circuitDiagonalize(T,RD,cl.zeroX_algorithm2)
   elif (method == 'cz') :
      circuitDiagonalize(T,RD,cl.zeroX_algorithm1_cz)
   elif (method == 'greedy1') :
      circuitDiagonalize(T,RD,cl.zeroX_algorithm1_greedy1)
   elif (method == 'greedy2') :
      circuitDiagonalize(T,RD,cl.zeroX_algorithm1_greedy2)
   elif (method == 'cnot') :
      circuitDiagonalize_CNot(T,RD,[])
   elif (method == 'cnot_log2') :
      circuitDiagonalize_CNot(T,RD,[int(math.ceil(math.log2(T.n)))])
   elif (method == 'cnot_best') :
      circuitDiagonalize_CNot(T,RD,range(1,T.n+1))
   else :
      raise Exception("Invalid method for simultaneous diagonalization (%s)" % method)

   # Generate the circuit
   circuitDiagExponentiateGenerate(R, RD, T, weights, funOrderZ)



# ======================================================================
# Circuit generation for direct exponentiation of Paulis
# ======================================================================

def circuitDirectExponentiation(R, paulis, weights, optimized=True) :
   # Send all gate operations to the given recorder R
   T = cl.Tableau(cl.pauli_to_matrix(paulis))
   C = T.getX() + 2 * T.getZ() # 0=I, 1=X, 2=Z, 3=Y
   w = (1 - 2 * T.getSign()) * weights
   m = T.m
   n = T.n

   G = [[],
        [R.addH],
        [],
        [R.addS, R.addH]]
   Gc= [[],
        [R.addH],
        [],
        [R.addH, R.addSdg]]

   if (optimized) :
      s = np.zeros(n,C.dtype)
      for i in range(m+1) :
         # Set the new state
         sNew = C[i,:] if (i < m) else np.zeros(n,C.dtype)

         # Add closing cnots if needed
         idx = list(np.where((s != 0) & (sNew != s))[0])
         idx.reverse()
         for j in idx :
            R.addCNot(j,n)

         # Add closing and open gate if needed
         idx = np.where(sNew != s)[0]
         for j in idx :
            for gateFun in Gc[s[j]] :
               gateFun(j)
            for gateFun in G[sNew[j]] :
               gateFun(j)

         # Add opening cnots if needed
         idx = np.where((sNew != 0) & (sNew != s))[0]
         for j in idx :
            R.addCNot(j,n)

         # Add rotation to the ancilla
         if (i < m) :
            # For each Y term we must negate the weight
            if (np.sum(C[i,:] == 3) % 2 == 1) :
               w[i] *= -1
            R.addRz(n,w[i])

         # Update the state
         s = sNew

   else :
      # Individual exponentiation of pauli terms
      for i in range(m) :
         p = C[i,:]
         # Add opening gates
         for j in range(n) :
            for gateFun in G[p[j]] :
               gateFun(j)

         # Add opening cnots
         idx = list(np.where(p != 0)[0])
         for j in idx :
            R.addCNot(j,n)

         # Add rotation to the ancilla; for each Y term we must negate the weight
         if (np.sum(C[i,:] == 3) % 2 == 1) :
            w[i] *= -1
         R.addRz(n, w[i])

         # Add opening cnots
         idx = list(np.where(p != 0)[0])
         idx.reverse()
         for j in idx :
            R.addCNot(j,n)

         # Add closing gates
         for j in range(n) :
            for gateFun in Gc[p[j]] :
               gateFun(j)



# ======================================================================
# Circuit generation for exponentiation of Paulis
# ======================================================================

def circuitExponentiate(R, paulis, weights, method) :
   method = method.lower()
   if (method[:5] == 'diag_') :
      circuitDiagExponentiation(R, paulis, weights, method=method[5:], funOrderZ=cl.orderZ)
   elif (method == 'direct') :
      circuitDirectExponentiation(R, paulis, weights, optimized=True)
   elif (method == 'direct_opt') :
      index   = cl.orderPaulis(cl.pauli_to_matrix(paulis))
      paulis  = [paulis[i] for i in index]
      weights = weights[index]
      circuitDirectExponentiation(R, paulis, weights, optimized=True)
   else :
      raise Exception("Invalid method for exponentiation (%s)" % method)
