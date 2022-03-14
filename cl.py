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

import numpy as np
import numpy.random as random
import os.path
import pickle
import time

from pathlib import Path


# Constants and look-up tables
binary_to_pauli = {(0,0) : 'i',
                   (1,0) : 'x',
                   (1,1) : 'y',
                   (0,1) : 'z'}

pauli_to_binary = {'i' : (0,0),
                   'I' : (0,0),
                   'x' : (1,0),
                   'X' : (1,0),
                   'y' : (1,1),
                   'Y' : (1,1),
                   'z' : (0,1),
                   'Z' : (0,1)}


pauli_to_array = {'i' : np.asarray([[1,0],[0,1]],dtype=np.complex),
                  'x' : np.asarray([[0,1],[1,0]],dtype=np.complex),
                  'y' : np.asarray([[0,-1j],[1j,0]],dtype=np.complex),
                  'z' : np.asarray([[1,0],[0,-1]],dtype=np.complex),
                  'j' : np.asarray([[1j]],dtype=np.complex),
                  '-' : np.asarray([[-1]],dtype=np.complex)}

pauli_prefix = ['', 'j', '-', '-j']

pauli_commute = {('i','i') : 0,
                 ('i','x') : 0,
                 ('i','y') : 0,
                 ('i','z') : 0,
                 ('x','i') : 0,
                 ('x','x') : 0,
                 ('x','y') : 1,
                 ('x','z') : 1,
                 ('y','i') : 0,
                 ('y','x') : 1,
                 ('y','y') : 0,
                 ('y','z') : 1,
                 ('z','i') : 0,
                 ('z','x') : 1,
                 ('z','y') : 1,
                 ('z','z') : 0}

matrix_dtype = np.int8



def paulis_commute(p1,p2) :
   return (sum([pauli_commute[k] for k in zip(p1,p2)]) % 2 == 0)


def random_commuting_matrix(n,k,flagSign=False, flagComplex=False, init=None) :
   dtype = np.int32
   k = min(k,2**n)

   R = np.zeros([k,2*n+2],dtype=dtype)
   if (init is not None) :
      R[:init.shape[0],:init.shape[1]] = init
      m = init.shape[0]
   else :
      m = 0

   P = R[:,:2*n]
   innerprod = 0
   while (m < k) :
      v = random.randint(0,2,size=[2*n],dtype=dtype)
      Pm = P[m,:]
      Pm[0:n] = v[n:]
      Pm[n:2*n]  = v[:n]
      for i in range(m) :
         if (np.array_equal(P[i,:],Pm)) :
            innerprod = 1
         else :
            innerprod = np.dot(P[i,:],v) % 2
         if (innerprod == 1) :
            break
      if (innerprod == 0) :
         m += 1

   if (flagSign) :
      R[:,2*n] = random.randint(0,2,size=m,dtype=dtype)
   if (flagComplex) :
      R[:,2*n+1] = random.randint(0,2,size=m,dtype=dtype)
   return R

def random_matrix(n,k,flagSign=False) :
   dtype = np.int32
   R = np.zeros([k,2*n+2],dtype=dtype)
   R[:,:2*n] = random.randint(0,2,size=(k,2*n),dtype=dtype)
   if (flagSign) :
      R[:,2*n:] = random.randint(0,2,size=(k,2),dtype=dtype)
   return R

def pauli_to_matrix(P) :
   flagImplicit = not isinstance(P,list)
   if (flagImplicit) :
     P = [P]

   valueExponent= []
   valuePaulis  = []

   n = 0
   for pauli in P :
      idx = 0

      # Initialize the exponent: each 'Y' term represents X*Z= jY
      #exponent = -1 * sum([int(term=='y') for term in pauli])
      exponent = 0

      # Get the sign
      if (pauli[idx] == '-') :
         exponent += 2; idx += 1

      # Check for complex
      if (pauli[idx] == 'j') :
         exponent += 1; idx += 1

      # Add the normalized exponent
      valueExponent.append(exponent % 4)

      # Convert the pauli string
      value  = [pauli_to_binary[term] for term in pauli[idx:]] # List of tuples
      valueX = [v[0] for v in value]
      valueY = [v[1] for v in value]
      valuePaulis.append(valueX + valueY)
      if (n == 0) :
         n = len(valuePaulis[-1])
      elif (n != len(valuePaulis[-1])) :
         raise Exception("Mismatch in pauli dimensions")

   # Create the matrix
   M = np.zeros((len(P), n+2),dtype=matrix_dtype)

   # Add the pauli part
   for (i,value) in enumerate(valuePaulis) :
      M[i,:n] = value

   # Add the binary representation of the exponent
   M[:,-2] = [v//2 for v in valueExponent]
   M[:,-1] = [v%2   for v in valueExponent]

   if (flagImplicit) : M = M[0,:]

   return M

def pauli_to_numpy(p) :
   M = np.asarray([[1]],dtype=np.complex)
   for idx in range(len(p)) :
      m = p[-1-idx]
      M = np.kron(M,pauli_to_array[m])
   return M


def matrix_to_pauli(M) :
   flagImplicit = (M.ndim == 1)
   if (flagImplicit) :
      M = M.reshape([1,M.size])

   S = []
   n = (M.shape[1] - 2) // 2

   for (i,m) in enumerate(M) :
      X = list(m[:n])
      Z = list(m[n:])
      pauli = ''.join([binary_to_pauli[key] for key in zip(X,Z)])
      exponent = 2*m[-2] + m[-1]
      S.append(pauli_prefix[exponent % 4] + pauli)

   if (flagImplicit) : S = S[0]
   return S

def multiply_rows(row1,row2,conjugate=False) :
   # Multiply row1 * row2
   n = (row1.size - 1) // 2

   # Determine exponent to j
   if (conjugate) :
      exponent = (2*row1[-2] + row1[-1]) + (2*row2[-2] + 3*row2[-1])
   else :
      exponent = (2*row1[-2] + row1[-1]) + (2*row2[-2] + row2[-1])

   for i in range(n) :
      x1 = row1[i]; z1 = row1[i+n]
      x2 = row2[i]; z2 = row2[i+n]
      if (x1 == 1) :
         if (z1 == 0) :
            exponent += z2*(2*x2 - 1) # Multiply by X
         else :
            exponent += z2 - x2       # Multiply by Y
      elif (z1 == 1) :
            exponent += x2*(1 - 2*z2) # Multiply by Z

   # Set the result
   exponent %= 4
   result = row1 ^ row2
   result[-2] = exponent//2
   result[-1] = exponent%2

   return result

def factorize(M) :
   (m,n) = M.shape

   # Allocated and initialize the basis
   basis = np.copy(M)

   # Allocate the coefficient matrix
   k = min(m,2*n - 1)
   coef = np.zeros((m,k),dtype=matrix_dtype)
   indices = [idx for idx in range(m)]

   # Gaussian elimination
   i = 0; j = 0
   while ((i < m) and (j < n-2)) :
     x = (basis[i:,j] != 0)
     if (not any(x)) :
        j += 1
        continue

     # Find the first non-zero entry and swap rows
     x = np.nonzero(x)[0]
     idx = int(x[0]) + i
     if (idx != i) :
        basis[[i,idx],:] = basis[[idx,i],:]
        coef[[i,idx],:]   = coef[[idx,i],:]
        (indices[i], indices[idx]) = (indices[idx], indices[i])

     # Mark the new basis vector
     coef[i,i] = 1

     # Sweep remaining rows
     pivot = basis[i,:]
     for idx in list(x[1:]) :
        basis[idx+i,:] = multiply_rows(basis[idx+i,:],pivot,conjugate=True)
        coef[idx+i,i] = 1

     # Update i and j
     i += 1; j += 1

   # Factorize the exponent part
   extra = 0
   if (i < m) :
     n2 = sum(basis[i:,-2])
     n1 = sum(basis[i:,-1])
     if ((n1 == m-i) and ((n2 == 0) or (n2 == m-i))) :
        # All +i or -i
        for k in range(i,m) :
           coef[k,i] = 1
           i += 1; extra += 1
     else :
        # We need a sign
        coef[i:,i] = basis[i:,-2]
        if (n1 != 0) :
           coef[i:,i+1] = basis[i:,-1]
        basis[i,-2] = 1; basis[i,-1] = 0; i += 1; extra += 1
        if (n1 != 0) :
            basis[i,-2] = 0; basis[i,-1] = 1; i += 1; extra += 1

   # Extract the coefficients
   T = np.zeros((m,i),dtype=matrix_dtype)
   T[indices,:] = coef[:,:i]
   coef = T

   # Extract the basis
   basis = np.copy(basis[:i,:])

   return (basis,coef,extra)


def reconstruct(basis, coef) :
   m = coef.shape[0]
   n = basis.shape[1]

   M = np.zeros((m,n),dtype=matrix_dtype)

   for i in range(m) :
      for j in range(coef.shape[1]) :
         if (coef[i,j]) :
            M[i,:] = multiply_rows(basis[j,:], M[i,:])

   return M




class OperatorCNot(object) :
   def __init__(self, index1, index2) :
      self.index1 = index1
      self.index2 = index2

   def apply(self, tableau) :
      tableau.cnot_(self.index1, self.index2)

   def sendTo(self, recorder) :
      recorder.addCNot(self.index1, self.index2)

   def inverse(self) :
      return OperatorCNot(self.index1, self.index2)

   def isinverse(self, other) :
      return ((isinstance(other, OperatorCNot)) and
              (other.index1 == self.index1) and
              (other.index2 == self.index2))


class OperatorSingle(object) :
   pass


class OperatorH(OperatorSingle) :
   def __init__(self, index) :
      self.index = index

   def apply(self, tableau) :
      tableau.H_(self.index)

   def sendTo(self, recorder) :
      recorder.addH(self.index)

   def inverse(self) :
      return OperatorH(self.index)

   def isinverse(self, other) :
      return (isinstance(other, OperatorH) and (other.index == self.index))


class OperatorS(OperatorSingle) :
   def __init__(self, index) :
      self.index = index

   def apply(self, tableau) :
      tableau.S_(self.index)

   def sendTo(self, recorder) :
      recorder.addS(self.index)

   def inverse(self) :
      return OperatorSdg(self.index)

   def isinverse(self, other) :
      return (isinstance(other, OperatorSdg) and (other.index == self.index))


class OperatorSdg(OperatorSingle) :
   def __init__(self, index) :
      self.index = index

   def apply(self, tableau) :
      tableau.Sdg_(self.index)

   def sendTo(self, recorder) :
      recorder.addSdg(self.index)

   def inverse(self) :
      return OperatorS(self.index)

   def isinverse(self, other) :
      return (isinstance(other, OperatorS) and (other.index == self.index))


class OperatorX(OperatorSingle) :
   def __init__(self, index) :
      self.index = index

   def apply(self, tableau) :
      tableau.X_(self.index)

   def sendTo(self, recorder) :
      recorder.addX(self.index)

   def inverse(self) :
      return OperatorX(self.index)

   def isinverse(self, other) :
      return (isinstance(other, OperatorX) and (other.index == self.index))


class OperatorY(OperatorSingle) :
   def __init__(self, index) :
      self.index = index

   def apply(self, tableau) :
      tableau.Y_(self.index)

   def sendTo(self, recorder) :
      recorder.addY(self.index)

   def inverse(self) :
      return OperatorY(self.index)

   def isinverse(self, other) :
      return (isinstance(other, OperatorY) and (other.index == self.index))


class OperatorZ(OperatorSingle) :
   def __init__(self, index) :
      self.index = index

   def apply(self, tableau) :
      tableau.Z_(self.index)

   def sendTo(self, recorder) :
      recorder.addZ(self.index)

   def inverse(self) :
      return OperatorZ(self.index)

   def isinverse(self, other) :
      return (isinstance(other, OperatorZ) and (other.index == self.index))


class OperatorRz(OperatorSingle) :
   def __init__(self, index, angle) :
      self.index = index
      self.angle = angle

   def apply(self, tableau) :
      raise Exception("Operator Rz cannot be applied to the table")

   def sendTo(self, recorder) :
      recorder.addRz(self.index, self.angle)

   def inverse(self) :
      return OperatorRz(self.index, -1 * self.angle)

   def isinverse(self, other) :
      return (isinstance(other, OperatorRz) and (other.index == self.index) and
              (math.abs((other.angle + self.angle) % (2*math.pi)) < 1e-14))


class Recorder(object) :
   def __init__(self) :
      pass


class RecordOperations(Recorder) :
   def __init__(self, nQubits) :
      super(RecordOperations,self).__init__()
      self.nQubits = nQubits
      self.reset()

   def reset(self) :
      self.operations = []

   def add(self,operation) :
      self.operations.append(operation)

   def addH(self,index) :
      self.operations.append(OperatorH(index))

   def addS(self,index) :
      self.operations.append(OperatorS(index))

   def addSdg(self,index) :
      self.operations.append(OperatorSdg(index))

   def addX(self,index) :
      self.operations.append(OperatorX(index))

   def addY(self,index) :
      self.operations.append(OperatorY(index))

   def addZ(self,index) :
      self.operations.append(OperatorZ(index))

   def addRz(self,index,angle) :
      self.operations.append(OperatorRz(index,angle))

   def addCNot(self,index1,index2) :
      self.operations.append(OperatorCNot(index1,index2))

   def inverse(self) :
      r = RecordOperations(self.nQubits)
      for idx in range(len(self.operations)-1,-1,-1) :
         r.add(self.operations[idx].inverse())
      return r

   def apply(self, tableau) :
      order = tableau.get_qubit_order()
      tableau.unshuffle_qubits()

      for op in self.operations :
         op.apply(tableau)

      tableau.shuffle_qubits(order)

   def sendTo(self, recorder) :
      for op in self.operations :
         op.sendTo(recorder)

   def simplify(self) :
      while (True) :
         changes = False
         qubit_queues = [[] for i in range(self.nQubits)]
         for (idx,op) in enumerate(self.operations) :
            iop = (idx,op)
            if (isinstance(op, OperatorCNot)) :
               qubit_queues[op.index1].append(iop)
               qubit_queues[op.index2].append(iop)
            else :
               qubit_queues[op.index].append(iop)

         # Find  patterns
         operations = [op for op in self.operations]
         for qubit in range(self.nQubits) :
            queue = qubit_queues[qubit]
            index = 0
            while (index < len(queue)) :
               # Check matching gates
               if (index >= 1) :
                  if (queue[index-1][1].isinverse(queue[index][1])) :
                     simplify = False
                     if (queue[index-1][0] == queue[index][0]-1) :
                        simplify = True
                     elif (isinstance(queue[index-1][1],OperatorSingle)) :
                        simplify = True
                     elif (isinstance(queue[index-1][1],OperatorCNot)) :
                         # We can only simplify non-adjacent cnot operators
                         # if there are no blocking gates on the other qubit
                         # for the control-qubit diagonal single-qubit operators
                         # are permitted, for the X-qubit no operations are
                         # allowed in between --- Not implemented yet.
                         simplify = False

                     # Cancel operations
                     if (simplify) :
                        operations[queue[index-1][0]] = None
                        operations[queue[index][0]] = None
                        changes = True

               # Check double S
               #if (index >= 1) :
               #   if ((isinstance(queue[index-1][1], OperatorS)) and
               #       (isinstance(queue[index  ][1], OperatorS))) :
               #      # Replace first S by Z, delete the second S
               #      operations[queue[index-1][0]] = OperatorZ(qubit)
               #      operations[queue[index  ][0]] = None
               #      changes = True

               # Check [S(a), CNOT(a,b), S(a)] pattern
               #if (index >= 2) :
               #   if ((isinstance(queue[index-2][1], OperatorS)) and
               #       (isinstance(queue[index-1][1], OperatorCNot)) and
               #       (isinstance(queue[index  ][1], OperatorS)) and
               #       (queue[index-1][1].index2 != qubit)) :
               #      # Replace first S by Z, delete the second S
               #      operations[queue[index-2][0]] = OperatorZ(qubit)
               #      operations[queue[index  ][0]] = None
               #      changes = True

               index += 1

         # Check for changes
         if (changes == False) :
            break

         # Update the operators
         self.operations = [op for op in operations if (op is not None)]

   def getStages(self) :
      # Alternating single-qubit / cnot stages
      stages       = [[]]
      qubit_stage  = [0] * self.nQubits

      for op in self.operations :
         if (isinstance(op, OperatorCNot)) :
            index1 = op.index1
            index2 = op.index2
            stage = 2 * (max(qubit_stage[index1], qubit_stage[index2]) // 2) + 1
            qubit_stage[index1] = stage
            qubit_stage[index2] = stage
         else  :
            stage = qubit_stage[op.index]
            if (stage % 2 == 1) :
               # Odd numbered stages are for CNot
               stage += 1
               qubit_stage[op.index] = stage

         if (stage >= len(stages)) :
            stages.append([])
         stages[stage].append(op)
      return stages

   def getCount(self) :
      n1 = 0
      n2 = 0
      for op in self.operations :
         if (isinstance(op, OperatorCNot)) :
            n2 += 1
         else :
            n1 += 1
      return (n1,n2)


class RecordCNots(Recorder) :
   def __init__(self) :
      super(RecordCNots,self).__init__()
      self.reset()

   def reset(self) :
      self.cnots = []

   def addH(self,index) :
      pass

   def addS(self,index) :
      pass

   def addSdg(self,index) :
      pass

   def addX(self,index) :
      pass

   def addY(self,index) :
      pass

   def addZ(self,index) :
      pass

   def addCNot(self,index1,index2) :
      self.cnots.append((index1,index2))


class RecordCount(Recorder) :
   def __init__(self, nQubits) :
      super(RecordCount,self).__init__()
      self.nQubits = nQubits
      self.reset()

   def reset(self) :
      self.H     = 0
      self.S     = 0
      self.Sdg   = 0
      self.X     = 0
      self.Y     = 0
      self.Z     = 0
      self.cnot  = 0
      self.unitary = 0
      self.isUnitary = [False for i in range(self.nQubits)]

   def addH(self,index) :
      self.H += 1
      if (not self.isUnitary[index]) :
         self.unitary += 1
         self.isUnitary[index] = True

   def addS(self,index) :
      self.S += 1
      if (not self.isUnitary[index]) :
         self.unitary += 1
         self.isUnitary[index] = True

   def addSdg(self,index) :
      self.Sdg += 1
      if (not self.isUnitary[index]) :
         self.unitary += 1
         self.isUnitary[index] = True

   def addX(self,index) :
      self.X += 1
      if (not self.isUnitary[index]) :
         self.unitary += 1
         self.isUnitary[index] = True

   def addY(self,index) :
      self.Y += 1
      if (not self.isUnitary[index]) :
         self.unitary += 1
         self.isUnitary[index] = True

   def addZ(self,index) :
      self.Z += 1
      if (not self.isUnitary[index]) :
         self.unitary += 1
         self.isUnitary[index] = True

   def addCNot(self,index1,index2) :
      self.cnot += 1



class Tableau(object) :
   def __init__(self, param=0) :
      self.T = 0
      self.n = 0
      self.qubit = []       # Qubit mapping
      self.recorders = []

      if (isinstance(param,int)) :
         n = param
         self.T = np.zeros((n, 2*(n)+2), dtype=matrix_dtype)
      elif (isinstance(param,Tableau)) :
         self.T = np.copy(param.T)
      elif (isinstance(param,np.ndarray)) :
         T = np.copy(param)
         self.T = self.reshape(T,(1,T.size)) if (T.ndim == 1) else T
         self.n = (self.T.shape[1] - 2) // 2
      else :
         # Assume string or list of strings
         T = pauli_to_matrix(param)
         self.T = self.reshape(T,(1,T.size)) if (T.ndim == 1) else T

      # Set the qubit mapping
      if (isinstance(param,Tableau)) :
         self.qubit = [v for v in param.qubit]
      else :
         self.qubit = [v for v in range(self.n)]
      self.m = self.T.shape[0]
      self.n = (self.T.shape[1] - 2) // 2


   def __str__(self) :
      T = self.T
      m = self.m
      n = self.n

      s = [[' '+str(T[i,j]) for j in range(T.shape[1])] for i in range(T.shape[0])]
      s = [''.join(v) for v in s]
      s = [('[' + v[1:2*n] + ']   [' + v[2*n+1:4*n] + ']   [' + v[4*n+1:] + ']') for v in s]

      idx = ' '.join(['|'] * n)
      idx = ' ' + idx + '     ' + idx
      s.insert(0,idx)
      s.append(idx)

      t = self.n; k = 1
      while (t >= 10) :
         t //= 10; k += 1

      v = [("%*d" % (-k,q)) for q in self.qubit]
      for i in range(k) :
         idx = ' '.join([w[i] for w in v])
         s.append(' ' + idx + '     ' + idx)

      v = [("%*d" % (k,q)) for q in range(self.n)]
      for i in range(k) :
         idx = ' '.join([w[i] for w in v])
         s.insert(i,' ' + idx + '     ' + idx)

      s.insert(0,'')
      return ('\n'.join(s))

   def __repr__(self) :
      return self.__str__()

   def addRecorder(self,r) :
      self.recorders.append(r)

   def getX(self) :
      return np.copy(self.T[:,:self.n])

   def getZ(self) :
      return np.copy(self.T[:,self.n:2*self.n])

   def getSignedZ(self) :
      return np.copy(self.T[:,self.n:2*self.n+1])

   def getSign(self) :
      return np.copy(self.T[:,-2])

   def assertXEmpty(self) :
      X = self.getX()
      if (X.sum() != 0) :
         raise Exception("The X block of the tableau is not empty")

   def cnot_(self, a, b) :
      if (a == b) :
          return
      T = self.T
      n = self.n
      xa = T[:,a]; za = T[:,a+n]
      xb = T[:,b]; zb = T[:,b+n]
      s  = T[:,2*n]

      T[:,2*n] = (s + (xa * zb * (xb+za+1))) % 2 # Update the sign
      T[:,b]   = (xb + xa) % 2 # Update xb
      T[:,a+n] = (za + zb) % 2 # Update za

   def S_(self, a) :
      T = self.T
      n = self.n
      xa = T[:,a];
      za = T[:,a+n]
      T[:,2*n] = (T[:,2*n] + (xa*za)) % 2 # Update the sign (X -> Y, Y -> -X)
      T[:,a+n] = (za + xa) % 2

   def X_(self, a) :
      T = self.T
      n = self.n
      za = T[:,a+n]
      T[:,2*n] = (T[:,2*n] + za) % 2 # Update the sign (Y,Z -> -Y,-Z)

   def Y_(self, a) :
      T = self.T
      n = self.n
      xa = T[:,a];
      za = T[:,a+n]
      T[:,2*n] = (T[:,2*n] + xa+za) % 2 # Update the sign (X,Z -> -X,-Z)

   def Z_(self, a) :
      T = self.T
      n = self.n
      xa = T[:,a];
      T[:,2*n] = (T[:,2*n] + xa) % 2 # Update the sign (X,Y -> -X,-Y)

   def Sdg_(self, a) :
      T = self.T
      n = self.n
      xa = T[:,a];
      za = T[:,a+n]
      T[:,2*n] = (T[:,2*n] + (xa*(1-za))) % 2 # Update the sign (X -> -Y, Y -> X)
      T[:,a+n] = (za + xa) % 2

   def H_(self, a) :
      T = self.T
      n = self.n
      T[:,2*n] = (T[:,2*n] + T[:,a]*T[:,a+n]) % 2
      T[:,[a,a+n]] = T[:,[a+n,a]]

   def cnot(self, a, b) :
      if (a == b) :
          return
      self.cnot_(a,b)
      for r in self.recorders :
         r.addCNot(self.qubit[a],self.qubit[b])

   def cz(self, a, b) :
      if (a == b) :
         return
      self.H(b)
      self.cnot(a,b)
      self.H(b)

   def H(self, a) :
      self.H_(a)
      for r in self.recorders :
         r.addH(self.qubit[a])

   def S(self, a) :
      self.S_(a)
      for r in self.recorders :
         r.addS(self.qubit[a])

   def Sdg(self, a) :
      self.Sdg_(a)
      for r in self.recorders :
         r.addSdg(self.qubit[a])

   def X(self, a) :
      self.X_(a)
      for r in self.recorders :
         r.addX(self.qubit[a])

   def Y(self, a) :
      self.Y_(a)
      for r in self.recorders :
         r.addY(self.qubit[a])

   def Z(self, a) :
      self.Z_(a)
      for r in self.recorders :
         r.addZ(self.qubit[a])

   def SH_(self, a) :
      self.S_(a)
      self.H_(a)

   def SH(self, a) :
      self.S(a)
      self.H(a)

   def rowswap(self, a, b) :
      if (a != b) :
         self.T[[a,b],:] = self.T[[b,a],:]

   def swap(self, a, b) :
      self.rowswap(a,b)

   def colswap(self, a,b) :
      T = self.T
      n = self.n

      if (a != b) :
         T[:,[a,b]] = T[:,[b,a]]
         T[:,[a+n,b+n]] = T[:,[b+n,a+n]]
         (self.qubit[a],self.qubit[b]) = (self.qubit[b],self.qubit[a])

   def sweep(self, a, b) :
      if (a != b) :
         self.T[a,:] = multiply_rows(self.T[a,:],self.T[b,:])

   def unshuffle_qubits(self) :
      n = self.n
      self.T[:,[idx   for idx in self.qubit]] = self.T[:,[idx   for idx in range(n)]]
      self.T[:,[idx+n for idx in self.qubit]] = self.T[:,[idx+n for idx in range(n)]]
      self.qubit = [idx for idx in range(n)]

   def shuffle_qubits(self, order) :
      n = self.n
      self.unshuffle_qubits()
      self.T[:,[idx   for idx in range(n)]] = self.T[:,[idx   for idx in order]]
      self.T[:,[idx+n for idx in range(n)]] = self.T[:,[idx+n for idx in order]]
      self.qubit = [v for v in order]

   def get_qubit_order(self) :
      return [idx for idx in self.qubit]



# ======================================================================
# Algorithm 1 for eliminating the elements in X
# ======================================================================

def zeroX_algorithm1_cz(tableau) :
   # Diagonalize X - possibly non-zero padding on right
   zeroX_algorithm1_step1(tableau)

   # Diagonalize X - Remove padding
   zeroX_algorithm1_step2(tableau)

   # Eliminate X block entries (cz decomposition)
   zeroX_algorithm1_step3_cz(tableau)


def zeroX_algorithm1_cnot(tableau) :
   zeroX_algorithm1_step1(tableau)
   zeroX_algorithm1_step2(tableau)
   zeroX_algorithm1_step3_cnot(tableau)

def zeroX_algorithm1_greedy1(tableau) :
   zeroX_algorithm1_step1(tableau)
   zeroX_algorithm1_step2(tableau)
   zeroX_algorithm1_step3_greedy1(tableau)

def zeroX_algorithm1_greedy2(tableau) :
   zeroX_algorithm1_step1(tableau)
   zeroX_algorithm1_step2(tableau)
   zeroX_algorithm1_step3_greedy2(tableau)


def zeroX_algorithm1_step1(tableau) :
   # Diagonalize as much as possible using only row/column swaps and H
   m = tableau.m
   n = tableau.n

   # Partially diagonalize the X and Z blocks
   k = 0
   for offset in [0,n] :
      for k in range(k,min(m,n)) :
         found = False
         for j in range(k,n) :
            idx = list(np.where(tableau.T[k:,offset+j])[0] + k)
            if (idx) :
               tableau.swap(k,idx[0])
               tableau.colswap(k,j)
               for i in list(np.where(tableau.T[:,offset+k])[0]) :
                   tableau.sweep(i,k)

               if (offset != 0) :
                  tableau.H(k)

               found = True
               break

         if (found == False) :
            break


def zeroX_algorithm1_step2(tableau) :
   # Eliminate remaining off-diagonal elements in the X block,
   # located next to the diagonal matrix
   m = tableau.m
   n = tableau.n
   cnots = []

   # Determine k
   k = 0
   while (k < min(m,n)) :
      if (tableau.T[k,k] == 0) : break
      k += 1

   for j in range(k,n) :
      cx = tableau.T[:,j]
      cz = tableau.T[:,j+n]

      # Method 0: Directly sweep column
      cnot = sum(cx); method = 0

      # Method 1: Exchange with Z
      nc = sum(cz);
      if (nc < cnot) : (cnot, method) = (nc, 1)

      # Method 2: Difference with Z
      nc = sum((cx + cz) % 2)
      if (nc < cnot) : (cnot, method) = (nc, 2)

      # Apply the best method
      if (method == 0) :
         pass
      elif (method == 1) :
         tableau.H(j)
      elif (method == 2) :
         tableau.S(j)
         tableau.H(j)
      else :
         raise Exception("Internal error: method %d is not implemented" % method)

      # Determine the required CNOT gates
      cnots += [(idx,j) for idx in list(np.where(tableau.T[:,j] != 0)[0])]

   # Apply all the cnot gates
   for (idx,j) in cnots :
      tableau.cnot(idx,j)


def zeroX_algorithm1_step3_cz(tableau) :
   m = tableau.m
   n = tableau.n
   k = sum([(tableau.T[i,i] != 0) for i in range(min(m,n))])

   for i in range(1,k) :
      for j in range(i) :
         if (tableau.T[i,j+n] == 1) :
            tableau.H(j)
            tableau.cnot(i,j)
            tableau.H(j)

   for i in range(k) :
      if (tableau.T[i,i+n] == 1) :
         tableau.SH(i)
      else :
         tableau.H(i)


def zeroX_algorithm1_step3_cnot(tableau) :
   # Decompose as H-S-CNOT-S-H
   m = tableau.m
   n = tableau.n

   for i in range(min(m,n)) :
      if (tableau.T[i,i] == 0) :
         break

      if (np.sum(tableau.T[i,n:i+n+1]) % 2 == 0) :
         tableau.S(i)

      # Sweep row up to diagonal
      for j in range(i) :
         if (tableau.T[i,j+n]) :
            tableau.cnot(i,j)
            tableau.sweep(i,j)

   for i in range(min(m,n)) :
      if (tableau.T[i,i]) :
         tableau.SH(i)


def zeroX_algorithm1_step3_greedy1(tableau) :
      m = tableau.m
      n = tableau.n

      # Determine the indices of the remaining diagonal entries
      indices = []
      for i in range(min(tableau.m,tableau.n)) :
         if (tableau.T[i,i]) :
             indices.append(i)

      while (len(indices) > 0) :
         # Consider sweeping of a single column
         c = m+n+1; ci = 0; cj = 0; flagPair = False
         for i in range(len(indices)) :
            idx = indices[i]
            s = np.sum(tableau.T[:,n+idx]) - int(tableau.T[idx,n+idx])
            if (s < c) :
               c = s; ci = i; cj = i;

         # Find two columns with the minimal distance
         for i in range(len(indices)) :
            idx1 = indices[i]
            v1 = np.copy(tableau.T[:,n+idx1]) # Copy the column

            for j in range(i+1,len(indices)) :
               idx2 = indices[j]
               v2 = np.copy(tableau.T[:,n+idx2]) # Copy the column
               v2[idx1] = v1[idx1]
               v2[idx2] = v1[idx2]
               s = np.sum((v1+v2)%2) + 1
               if (s < c) :
                  c = s; ci = i; cj = j; flagPair = True

         # Get the column indices and delete the second entry
         idx1 = indices[ci]; idx2 = indices[cj]; del indices[ci]

         if (flagPair) :
            # Maximize the similarity (aside from the diagonal)
            if (tableau.T[idx2,n+idx1] != tableau.T[idx2,n+idx2]) :
               tableau.S(idx2)

            # Sweep column idx1 with column idx2 in the Z block
            tableau.cnot(idx1,idx2)

            # Eliminate fill-in with X
            tableau.sweep(idx1,idx2)

         # Eliminate the remaining elements in column idx1
         for i in range(m) :
            if ((i != idx1) and (tableau.T[i,n+idx1])) :
               tableau.cz(i,idx1)

         # Eliminate the diagonal entry in X
         if (tableau.T[idx1,n+idx1]) :
            tableau.SH(idx1)
         else :
            tableau.H(idx1)


def zeroX_algorithm1_step3_greedy2(tableau) :
      m = tableau.m
      n = tableau.n

      # Determine the indices of the remaining diagonal entries
      indices = []
      for i in range(min(tableau.m,tableau.n)) :
         if (tableau.T[i,i]) :
             indices.append(i)

      while (len(indices) > 0) :
         # Find two columns with the minimal distance
         bestCNot   = n+m
         bestSingle = 0
         bestIdx1   = None
         bestIdx2   = None

         # Consider sweeping of a single column
         for i in range(len(indices)) :
            idx    = indices[i]
            cnot   = np.sum(tableau.T[:,n+idx]) - int(tableau.T[idx,n+idx])
            single = 2 if (tableau.T[idx,n+idx]) else 1 # S,H if 1, H only if 0
            single+= 2* cnot # cz = H,CX,H
            if (cnot <= bestCNot) :
               if ((cnot < bestCNot) or (single < bestSingle)) :
                  bestCNot   = cnot
                  bestSingle = single
                  bestIdx1   = i

         # Consider sweeping the column with another first
         for i in range(len(indices)) :
            idx1 = indices[i]
            v1 = np.copy(tableau.T[:,n+idx1]) # Copy the column

            for j in range(len(indices)) :
               if (i == j) :
                  continue

               # Resulting column
               idx2 = indices[j]
               v = (v1 + tableau.T[:,n+idx2]) % 2

               # Count the operations
               cz = np.sum(v); single = 0; flip = False

               if (v[idx1]) :
                  # Sweeping the diagonal entry does not require a cz operation
                  cz -= 1

               if (v[idx2]) :
                  # We can flip the entry in the second column if needed
                  cz     -= 1
                  single += 1 # Cancel the entry before cnot
                  flip    = True

               # Set the number of cnot operations
               cnot    = cz + 1
               single += 2*cz

               # Number of operations to clear the X entry
               single += 2 if (v[idx1]) else 1 # S,H if 1, H only if 0

               if (cnot <= bestCNot) :
                  if ((cnot < bestCNot) or (single < bestSingle)) :
                     bestCNot   = cnot
                     bestSingle = single
                     bestIdx1   = i
                     bestIdx2   = j

         # Get the column indices and delete the second entry
         idx1 = indices[bestIdx1]
         idx2 = indices[bestIdx2] if bestIdx2 else None
         del indices[bestIdx1]

         if (idx2) :
            # Maximize the similarity (aside from the diagonal)
            if (tableau.T[idx2,n+idx1] != tableau.T[idx2,n+idx2]) :
               tableau.S(idx2)

            # Sweep column idx1 with column idx2 in the Z block
            tableau.cnot(idx1,idx2)

            # Eliminate fill-in with X
            tableau.sweep(idx1,idx2)

         # Eliminate the remaining elements in column idx2
         for i in range(m) :
            if ((i != idx1) and (tableau.T[i,n+idx1])) :
               tableau.cz(i,idx1)

         # Eliminate the diagonal entry in X
         if (tableau.T[idx1,n+idx1]) :
            tableau.SH(idx1)
         else :
            tableau.H(idx1)


# ======================================================================
# Algorithm 2 for eliminating the elements in X, see
# Hector J. Garcia, Igor L. Markov, and Andrew W. Cross, "Efficient
# inner-product algorithm for stabilizer states", arXiv:1210.6646, 2013
# ======================================================================

def zeroX_algorithm2(tableau) :
   zeroX_algorithm2_step1(tableau)
   zeroX_algorithm2_step2(tableau)
   zeroX_algorithm2_step3(tableau)
   zeroX_algorithm2_step4(tableau)
   zeroX_algorithm2_step5(tableau)


def zeroX_algorithm2_step1(tableau) :
   # Normalize to echelon form
   m = tableau.m
   n = tableau.n
   i = 0
   for j in range(2*n) :
      idx = list(np.where(tableau.T[i:,j])[0] + i)
      if (idx) :
         tableau.rowswap(i,idx[0])
         for k in (idx[1:] + list(np.where(tableau.T[:i,j])[0])) :
            tableau.sweep(k, i)
         i += 1;
         if (i == m) : break

def zeroX_algorithm2_step2(tableau) :
   # Lines 3-19
   m = tableau.m
   n = tableau.n
   i = 0
   for j in range(n) :
      k = list(np.where(tableau.T[i:,j])[0] + i)
      if (k) :
         tableau.swap(i,k[0])
      else :
         k2 = list(np.where(tableau.T[i:,j+n])[0] + i)
         if (k2) :
            k2 = k2[-1]

            # Add sweeping
            for idx in range(i,k2) :
               if (tableau.T[idx,j+n]) :
                  tableau.sweep(idx,k2)
            tableau.swap(i,k2) # Last row

            if ((np.sum(tableau.T[i,j+1:n]) + np.sum(tableau.T[i,n+j+1:2*n])) != 0) :
               tableau.H(j)

      i += 1
      if (i == m) : break


def zeroX_algorithm2_step2_bug(tableau) :
   # Lines 3-19
   m = tableau.m
   n = tableau.n
   i = 0
   for j in range(n) :
      k = list(np.where(tableau.T[i:,j])[0] + i)
      if (k) :
         tableau.swap(i,k[0])
      else :
         k2 = list(np.where(tableau.T[i:,j+n])[0] + i)
         if (k2) :
            k2 = k2[-1]
            print(tableau)
            print("[i=%d, j=%d] using k2=%d" % (i,j,k2))
            # Necessary step: add sweeping
            #for idx in range(i,k2) :
            #   if (tableau.T[idx,j+n]) :
            #      tableau.sweep(idx,k2)
            tableau.swap(i,k2) # Last row
            print(tableau)
            input("Press <return> to continue . . .")
            if ((np.sum(tableau.T[i,j+1:n]) + np.sum(tableau.T[i,n+j+1:2*n])) != 0) :
               tableau.H(j)

      i += 1
      if (i == m) : break


def zeroX_algorithm2_step3(tableau) :
   # Lines 20-27
   m = tableau.m
   n = tableau.n
   for j in range(m) :
      for k in range(j+1,n) :
         if ((tableau.T[j,k] == 1)) :
            tableau.cnot(j,k)

def zeroX_algorithm2_step4(tableau) :
   # Lines 28-35
   m = tableau.m
   n = tableau.n
   for j in range(m) :
      for k in range(j+1,n) :
         if ((tableau.T[j,k]==0) and (tableau.T[j,k+n]==1)) :
            tableau.cz(j,k)

def zeroX_algorithm2_step5(tableau) :
   # Lines 36-47
   m = tableau.m
   n = tableau.n
   for j in range(min(m,n)) :
      if (tableau.T[j,j]) :
         if (tableau.T[j,j+n]) :
            tableau.SH(j)
         else :
            tableau.H(j)


# ======================================================================
# Algorithm 3a - O. Crawford, B. van Straaten, D. Wang, T. Parks,
# E. Campbell, and S. Brierley, Efficient quantum measurment of Pauli
# operators in the presence of finite sampling error. arXiv:1908.06942.
# ======================================================================

def zeroX_algorithm3a(tableau) :
   zeroX_algorithm3_stageA(tableau)
   zeroX_algorithm3_stageB(tableau)

def zeroX_algorithm3b(tableau) :
   k = zeroX_algorithm3_stageA(tableau)
   zeroX_algorithm3_stageC(tableau, k)


def zeroX_algorithm3_stageA(tableau) :
   m = tableau.m
   n = tableau.n

   k = 0; i = 0

   while (i < m) :
      found = False
      for offset in [0,n] :
         for j in range(k,n) :
            if (tableau.T[i,j+offset] == 1) :
               # Move to column k
               if (offset != 0) :
                  tableau.H(j)
               if (j > k) :
                  tableau.colswap(k,j)

               # Sweep the remaining columns
               for ell in range(m) :
                  if (ell != i) and (tableau.T[ell,k] == 1) :
                     tableau.sweep(ell, i)

               k = k + 1; found = True
               break

         if (found) :
            break

      # Deal with row that are in the range of the previous rows
      if (not found):
         if (i != m-1) :
            tableau.rowswap(i,m-1)
         m = m - 1
      else :
         i = i + 1

   # We should now have a tableau in which the top-left k-by-k block
   # is the identity matrix, any remaining rows are zero.

   # Step 2 - Augment the matrix if needed
   if (m < n) :
      T = np.zeros((n,2*n+2),dtype=tableau.T.dtype)
      T[:m,:] = tableau.T[:m,:]
      T[m:n,m:n] = np.eye(n-m,dtype=T.dtype)
      T[m:n,n:n+m] = tableau.T[:m,n+m:2*n].T
      tableau.T = T
      tableau.m = n
      tableau.n = n

   # Step 3 - Sweep remaining columns
   for i in range(m,n) :
      for ell in range(n) :
         if (ell != i) and (tableau.T[ell,i] == 1) :
            tableau.sweep(ell, i)

   return k


def zeroX_algorithm3_stageB(tableau) :
   m = tableau.m
   n = tableau.n

   # Step 4 - Clear off-diagonal entries in Z
   for i in range(n) :
      for j in range(i+1,n) :
         if (tableau.T[i,j+n]) :
            tableau.cz(i,j)

   # Step 5 - Clear the X block
   for i in range(n) :
      if (tableau.T[i,i+n]) :
         tableau.SH(i)
      else :
         tableau.H(i)


def zeroX_algorithm3_stageC(tableau, k) :
   m = tableau.m
   n = tableau.n

   # Step 4 - Cholesky
   E = tableau.T[:k,n:n+k]
   [M,D] = decomposeSymmetric(E)

   for i in range(k) :
      if (D[i]) :
         tableau.S(i)


   # Sweep using CNot operations
   for i in range(k) :
      # Make sure the diagonal entry is set (may not be needed)
      if (tableau.T[i,i+n] == 0) :
         tableau.S(i)

      for j in range(i+1,k) :
         if (tableau.T[i,j+n]) :
           tableau.cnot(j,i)
           tableau.sweep(j,i)

   # Add diagonal entries
   for i in range(k,n) :
      tableau.S(i)

   # Full Cholesky
   E = tableau.T[:n,n:2*n]
   [M,D] = decomposeSymmetric(E)
   for i in range(n) :
      if (D[i]) :
         tableau.S(i)

   # Sweep using CNot operations
   for i in range(n) :
      # Make sure the diagonal entry is set (may not be needed)
      if (tableau.T[i,i+n] == 0) :
         tableau.S(i)

      for j in range(i+1,n) :
         if (tableau.T[i,j+n]) :
           tableau.cnot(j,i)
           tableau.sweep(j,i)

   # Finalize the tableau
   for i in range(n) :
      tableau.SH(i)



def decomposeSymmetric(A) :
   # Use Lemma 7 in S. Aaronson and D. Gottesman, Improved simulation
   # of stabilizer states, Phys. Rev. A 70, 052328 (2004).
   n = A.shape[0]
   M = np.zeros((n,n),dtype=A.dtype)
   D = np.zeros(n, dtype=A.dtype)

   print(A.shape)

   for i in range(n) :
      for j in range(0,i) :
         # Compute element M[i,j]
         M[i,j] = (A[i,j] + np.sum(M[i,:j] * M[j,:j])) % 2

      # Set the diagonal element
      M[i,i] = 1

      # Compute D
      D[i] = (A[i,i] + np.sum(M[i,:i+1])) % 2

   return (M,D)


def invertLowerDiag(A) :
   n = A.shape[0]
   M = np.eye(n,dtype=A.dtype)

   for i in range(n) :
      for j in range(i-1,-1,-1) :
         M[i,j] = (np.sum(M[i,j+1:i+1] * A[j+1:i+1,j])) % 2


   return M

# ======================================================================
# Normalize the tableau
# ======================================================================

def normalizeTableau_step1(tableau) :
   # Diagonalize as much as possible
   m = tableau.m
   n = tableau.n
   idxX    = []
   idxZ    = []
   idxNone = []

   # Partially diagonalize the X and Z blocks
   k = 0
   for k in range(k,min(m,n)) :
      found = False
      for offset in [0,n] :
         idx = list(np.where(tableau.T[k:,offset+k])[0] + k)
         if (idx) :
            tableau.swap(k,idx[0])

            for i in list(np.where(tableau.T[:,offset+k])[0]) :
               tableau.sweep(i,k)

            if (offset != 0) :
               idxZ.append(k)
            else :
               idxX.append(k)

            found = True
            break

      if (not found) :
         idxNone.append(k)

   return (idxX, idxZ, idxNone)



# ======================================================================
# Test problem generation
# ======================================================================

template = 'cache/basis_%d_%d.npy'

def generate_test_basis(n,idx,verbose=False) :
   filename = template % (n,idx)
   if (os.path.isfile(filename)) :
      basis = np.load(filename)
      return basis

   # Generate an initial matrix
   if (verbose) :
      print("[%d,%d] Creating initial matrix . . ." % (n,idx))

   # Store to original random state
   state = np.random.get_state()

   np.random.seed(1001 * n + idx)
   M = random_commuting_matrix(n,1,flagSign=True, flagComplex=False, init=None)

   while True :
      # Decompose
      (basis,coef,extra) = factorize(M)
      if (basis.shape[0]-extra == n) :
         break

      # Append a new element
      if (verbose) :
         print("[%d,%d] Expanding to size %d . . ." % (n,idx,M.shape[0]+1))
      M = random_commuting_matrix(n,M.shape[0]+1,flagSign=True, flagComplex=False, init=M)

   basis = basis[:basis.shape[0]-extra,:]
   np.save(filename, basis)

   # Restore to original random state
   np.random.set_state(state)

   return basis


def weight_matrix_is_full_rank(W) :
   (m,n) = W.shape
   W = np.copy(W)

   # Sweep the matrix to diagonal
   i = 0; k = m
   while ((i < k) and (i < n)) :
      idx = list(np.where(W[i,:])[0])
      if (len(idx) == 0) :
         k -= 1
         W[[i,k],:] = W[[k,i],:] # Swap the rows
         continue
      else :
         idx = idx[0]
         W[:,[i,idx]] = W[:,[idx,i]] # Swap the columns
         for r in list(np.where(W[:,i])[0]) :
            if (r == i) : continue
            W[r,:] = (W[r,:] + W[i,:]) % 2
         i += 1

   return (i == min(m,n))


def generate_full_rank_weights(m,n,p=0.5,seed=None) :
   # Store to original random state
   if (seed is not None) :
      state = np.random.get_state()
      np.random.seed(seed)

   while True :
     # Generate a random set of binary weights
     W = 1 * (np.random.random((m,n)) < p)
     if (weight_matrix_is_full_rank(W)) :
        break

   # Restore to original random state
   if (seed is not None) :
      np.random.set_state(state)

   return W

def generate_permutation(n) :
   return np.argsort(np.random.random(n))


def create_basic_problem(n, index) :
    B = generate_test_basis(n,index)
    C = generate_full_rank_weights(n,n,seed=100*n+index)
    M = np.dot(C,B) % 2
    return M


# ======================================================================
# Reduction of the CNot operations as described in:
# Ketan N. Patel, Igor L. Markov, and John P. Hayes, "Optimal synthesis
# of linear reversible circuits", Quantum Information and Computation,
# Vol. 8, No. 3&4, pp.282--294, 2008
# ======================================================================

def get_cnot_matrix(cnots, n) :
   # List of cnot(a,b) of the form [(a,b),...]
   M = np.eye(n,dtype=np.int16)
   for (a,b) in cnots :
      M[b,:] = (M[b,:] + M[a,:]) % 2
   return M

def ensure_cnot_equivalence(c1,c2,n) :
   M1 = get_cnot_matrix(c1,n)
   M2 = get_cnot_matrix(c2,n)
   if (np.abs(M1-M2).sum() != 0) :
      raise Exception("The cnot operations are not equivalent!")

def reduce_cnot(cnots, n, blocksize) :
   # List of cnot(a,b) of the form [(a,b),...]
   M = get_cnot_matrix(cnots, n)

   def _sweep(a,b, M, operations, transpose) :
      # Sweep a with b
      M[a,:] = (M[a,:] + M[b,:]) % 2
      if (transpose) :
         operations[1].append((a,b))
      else :
         operations[0].append((b,a))


   # Apply the sweep operation
   operations = [[],[]]
   for transpose in [False,True] :
      offset = 0

      while (offset < n) :
         # Determine the block size
         size = min(blocksize, n-offset)

         # Template sweep
         for i in range(offset+size, n) :
            v = M[i,offset:offset+size]
            for j in range(offset,offset+size) :
               if (np.all(v == M[j,offset:offset+size])) :
                  _sweep(i,j, M, operations, transpose)
                  break

         # Diagonalize block
         for j in range(offset,offset+size) :
            for i in range(j+1,offset+size) :
               if (M[i,j] != 0) :
                 _sweep(i,j, M, operations, transpose)

         # Sweep remaining sub-block entries
         for j in range(offset,offset+size) :
            for i in range(offset+size,n) :
               if (M[i,j] != 0) :
                 _sweep(i,j, M, operations, transpose)

         # Update the offset
         offset += size

      # Transpose M
      M = M.T

   operations[0].reverse()
   return operations[1] + operations[0]



# ======================================================================
# Ordering of terms based on Z matrix
# ======================================================================

def orderZ(Z) :
   def _orderZ(indices,direction,depth) :
      if (depth == 0) :
         return indices

      idx0 = []
      idx1 = []
      for i in indices :
         if (Z[i,depth] == 0) :
            idx0.append(i)
         else :
            idx1.append(i)

      if (len(idx0)) : idx0 = _orderZ(idx0,direction,depth+1)
      if (len(idx1)) : idx1 = _orderZ(idx1,1-direction,depth+1)
      if (direction == 0) :
         return idx0 + idx1
      else :
         return idx1 + idx0

   index = [x for x in range(Z.shape[0])]
   depth = -1 * Z.shape[1]
   index = _orderZ(index,0,depth)
   return index


def countCNot(Z) :
   (m,n) = Z.shape
   z = np.zeros(n,matrix_dtype)
   v = z; op2 = 0
   for i in range(m) :
      op2 += np.sum((v != Z[i,:]))
      v = Z[i,:]
   op2 += np.sum((v != z))
   return op2


def orderZ_randomized(Z,n) :
   bestIndex = list(range(Z.shape[0]))
   bestCount = countCNot(Z[bestIndex,:])

   for i in range(n) :
      idx   = generate_permutation(Z.shape[1])
      index = orderZ(Z[:,idx])
      count = countCNot(Z[index,:])
      if (count < bestCount) :
         bestCount = count
         bestIndex = index

   return bestIndex


# ======================================================================
# Ordering of terms based on Paulis
# ======================================================================

def orderPaulis(M) :
   _nCNots  = np.asarray([0,1,1,1],dtype=np.int16)
   _nSingle = np.asarray([0,0,1,2],dtype=np.int16)

   T = Tableau(M)
   C = T.getX() + 2 * T.getZ()
   s = np.zeros(T.n,dtype=matrix_dtype)
   index = list(range(T.m))
   order = []

   while (index) :
      # Find the term best matching the current state
      bestIndex = 0
      bestCNot  = T.n+1
      bestSingle= 0
      bestState = None

      for j in range(len(index)) :
         # Determine the cost
         sNew = C[index[j],:]
         idx = np.where(s != sNew)[0]
         cnot = np.sum(_nCNots[sNew[idx]])
         if (cnot <= bestCNot) :
            single = np.sum(_nSingle[sNew[idx]])
            if ((cnot < bestCNot) or (single < bestSingle)) :
               bestIndex = j
               bestCNot = cnot
               bestSingle = single
               bestState = sNew

      # Select index, update state
      order.append(index[bestIndex])
      del index[bestIndex]
      s = sNew

   # Validate result
   if ((len(order) != T.m) or (len(set(order)) != T.m)) :
      raise Exception("Invalid reordering detected in function orderPaulis")

   return order


def countOpsPauliExp(M) :
   _nCNots  = np.asarray([0,1,1,1],dtype=np.int16)
   _nSingle = np.asarray([0,0,1,2],dtype=np.int16)

   T = Tableau(M)
   C = T.getX() + 2 * T.getZ()
   s = np.zeros(T.n,dtype=matrix_dtype)

   cnot = 0
   single = 0

   for i in range(T.m+1) :
      if (i < T.m) :
         sNew = C[i,:]
      else :
         sNew = np.zeros(T.n,dtype=matrix_dtype)

      # Determine the cost
      idx = np.where(s != sNew)[0]
      cnot += np.sum(_nCNots[sNew[idx]])
      single += np.sum(_nSingle[s[idx]]) + np.sum(_nSingle[sNew[idx]])

      # Update the state
      s = sNew

   return (cnot, single)


def orderPaulis_randomized(M,n) :
   #bestIndex = list(range(M.shape[0]))
   #(bestCNot, bestSingle) = countOpsPauliExp(Z[bestIndex,:])
   bestIndex = orderPaulis(M)
   (bestCNot, bestSingle) = countOpsPauliExp(M[bestIndex,:])

   for i in range(n) :
      index1 = generate_permutation(M.shape[0])
      Mbar = M[index1,:]
      index2 = orderPaulis(Mbar)
      (cnot, single) = countOpsPauliExp(Mbar[index2,:])

      if (cnot <= bestCNot) :
         if ((cnot < bestCNot) or (single < bestSingle)) :
            bestCNot   = cnot
            bestSingle = single
            bestIndex  = index1[index2]

   return list(bestIndex)


# ======================================================================
# Result cache
# ======================================================================

def ensureDirExists(directory) :
   try :
      os.makedirs(directory)
   except FileExistsError :
      pass

class ResultCache(object) :
   def __init__(self, name, cachedir='cache',timeout=60) :
      # Ensure the directory exists
      self.cachedir = cachedir
      ensureDirExists(self.cachedir)

      # Normalize the name
      self.name = name
      self.filename = name.replace(' ','_')
      self.filename = self.filename.replace('/','')
      self.filename = self.filename.replace('\\','')
      self.filename = self.filename.replace('.','')
      self.filename = "cache_" + self.filename + '.dat'

      # Load the current file
      self.load()

      # Set the modified flag to false
      self.modified     = False
      self.timesaved    = time.time()
      self.timeout      = timeout # Save every <timeout> seconds, if modified
      self.disable_save = False

   def load(self) :
      try :
         with open(Path(self.cachedir,self.filename),"rb") as fp :
            self.data = pickle.load(fp)
      except FileNotFoundError:
         self.data = {}

   def save(self) :
      if (not self.disable_save) :
         print("Saving %s . . ." % self.filename)
         with open(Path(self.cachedir,self.filename),"wb") as fp :
            pickle.dump(self.data, fp)
         self.timesaved = time.time()

   def flush(self) :
      if (self.modified) :
         self.save()

   def update(self) :
      if ((time.time() - self.timesaved) > self.timeout) :
         self.save()
         self.modified = False
      else :
         self.modified = True

   def __getitem__(self, key) :
      return self.data[key]

   def __setitem__(self, key, value) :
      self.data[key] = value
      self.update()

   def __delitem__(self, key) :
      try :
         del self.data[key]
         self.update()
      except KeyError :
         pass

   def __contains__(self, key) :
      return key in self.data
