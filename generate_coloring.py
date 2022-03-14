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

import cl_chemistry as clchem
from pathlib import Path

#strategies = ['largest_first','random_sequential','smallest_last','independent_set',
#              'connected_sequential_bfs','connected_sequential_dfs','connected_sequential',
#              'saturation_largest_first']

strategies = ['largest_first','independent_set', 'sequential']

entries = clchem.index_molecules()
entries.sort()

for entry in entries :
   for instance in entry[1] :
      instance = instance[1]
      paulis = instance.Hamiltonian()

      for strategy in strategies :
         if ((len(paulis) > 40000) and (strategy not in ['sequential','largest_first'])) :
            continue
         if ((len(paulis) > 100000) and (strategy not in ['sequential'])) :
            continue

         partition = instance.coloring(strategy,cache=True,compute=True)
