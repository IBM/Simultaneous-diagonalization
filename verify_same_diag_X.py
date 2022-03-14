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
import numpy as np
import time

P = None

n = 8; index = 6
M = cl.create_basic_problem(n,index)

print("-----------------------------------------------")
print("Test #1")
print("-----------------------------------------------")
for index2 in range(1000) :
   m = 20
   C = cl.generate_full_rank_weights(m,n,seed=index2)
   Mbar = np.dot(C,M) % 2
   T = cl.Tableau(Mbar)
   cl.zeroX_algorithm1_step1(T)
   if ((P is None) or (not np.all(P == T.T[:,:2*n]))) :
      print(T)
      P = T.T[:,:2*n]

print("")
print("-----------------------------------------------")
print("Test #2")
print("-----------------------------------------------")
P = None
for index2 in range(1000) :
   m = 20
   C = cl.generate_full_rank_weights(m,n,seed=index2)
   Mbar = np.dot(C,M) % 2
   T = cl.Tableau(Mbar)
   (a,b,c) = cl.normalizeTableau_step1(T)
   if ((P is None) or (not np.all(P == T.T[:,:2*n]))) :
      print(T)
      P = T.T[:,:2*n]


print("")
print("-----------------------------------------------")
print("Test #3")
print("-----------------------------------------------")
P = None
C = cl.generate_full_rank_weights(4,n,seed=index2)
M = np.dot(C,M) % 2

for index2 in range(10) :
   m = 10
   C = cl.generate_full_rank_weights(m,4,seed=index2)
   Mbar = np.dot(C,M) % 2
   T = cl.Tableau(Mbar)
   (a,b,c) = cl.normalizeTableau_step1(T)
   if ((P is None) or (not np.all(P == T.T[:,:2*n]))) :
      print(T)
      P = T.T[:,:2*n]
