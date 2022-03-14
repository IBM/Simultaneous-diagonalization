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

M = create_basic_problem(6,12)


T = Tableau(M)
R = RecordCount(T.n)
T.addRecorder(R)
zeroX_algorithm1_cz(T)
print((R.cnot,R.unitary))


T = Tableau(M)
R = RecordCount(T.n)
T.addRecorder(R)
zeroX_algorithm1_greedy1(T)
print((R.cnot,R.unitary))


T = Tableau(M)
R = RecordCount(T.n)
T.addRecorder(R)
zeroX_algorithm1_greedy2(T)
print((R.cnot,R.unitary))
