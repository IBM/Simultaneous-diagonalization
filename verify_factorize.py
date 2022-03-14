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

import numpy, time
np.random.seed(int(time.time()))

for i in [1] : #  range(1000) :
   np.random.seed(i)

   P = cl.random_commuting_matrix(3,6,True,False)
   print(P)
   print(cl.matrix_to_pauli(P))

   (basis,coef,extra) = cl.factorize(P)
   print("\nBasis(%+d)" % extra)
   print(basis)
   print(cl.matrix_to_pauli(basis))
   print("\nCoefficients")
   print(coef)

   Pbar = cl.reconstruct(basis,coef)
   print(cl.matrix_to_pauli(cl.reconstruct(basis[:-1,:],coef[:,:-1])))

   s = np.sum(np.abs(Pbar - P))
   print((extra, s))
   if (s != 0) or (extra != 2):
     print(i)
     break
