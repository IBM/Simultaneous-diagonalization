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
import math

total_error = 0

for n in range(5,26) :
   for index in range(20) :
      B = generate_test_basis(n,index)
      C = generate_full_rank_weights(n,n,seed=100*n+index)
      M = np.dot(C,B) % 2

      T = Tableau(M)
      R = RecordCNots()
      T.addRecorder(R)

      zeroX_algorithm1_cnot(T)

      optimal_blocksize = 0
      optimal_cnots     = []
      log_blocksize     = min(math.ceil(math.log2(n)),T.n)
      log_cnots         = 0

      for blocksize in range(1,T.n) :
         cnots = reduce_cnot(R.cnots, T.n, blocksize)
         if ((len(cnots) < len(optimal_cnots)) or (blocksize == 1)) :
            optimal_blocksize = blocksize
            optimal_cnots     = cnots
         if (blocksize == log_blocksize) :
            log_cnots = len(cnots)

      cnots = optimal_cnots
      err = np.linalg.norm(get_cnot_matrix(R.cnots, T.n) - get_cnot_matrix(cnots, T.n),'fro')
      total_error += err
      print("%2d [%2d] Original = %3d, new = %3d, log = %3d, blocksize = %2d (%2d), error = %s" %
            (n,index,len(R.cnots),len(cnots),log_cnots, optimal_blocksize, log_blocksize, err))

print("TOTAL ERROR = %s" % total_error)
