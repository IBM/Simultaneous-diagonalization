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

# Diagonalization of Y using S,H only (negates sign)

import qiskit
import numpy as np

c1 = qiskit.QuantumCircuit(4)
c1.s(0); c1.h(0); c1.x(0); c1.cx(0,3) # Y
c1.cx(1,3) # Z
c1.s(2); c1.h(2); c1.x(2); c1.cx(2,3) # Y

alpha = 0.3
c1.rz(-alpha,3)
c1.x(3)
c1.rz(alpha,3)
c1.x(3)

c1.cx(2,3); c1.x(2); c1.h(2); c1.sdg(2)
c1.cz(1,3)
c1.cx(0,3); c1.x(0); c1.h(0); c1.sdg(0)

###

c2 = qiskit.QuantumCircuit(4)
c2.s(0); c2.h(0); c2.cx(0,3); # Y
c2.cx(1,3) # Z
c2.s(2); c2.h(2); c2.x(2); c2.cx(2,3) # Y

alpha = 0.3
c2.rz(-1 * -alpha,3)
c2.x(3)
c2.rz(-1 * alpha,3)
c2.x(3)

c2.cx(2,3); c2.x(2); c2.h(2); c2.sdg(2)
c2.cz(1,3)
c2.cx(0,3); c2.h(0); c2.sdg(0)

backend = qiskit.Aer.get_backend('unitary_simulator')
U1 = qiskit.execute(c1, backend).result().get_unitary()
U2 = qiskit.execute(c2, backend).result().get_unitary()

print(np.linalg.norm(U1[:8,:8]-U2[:8,:8],'fro'))
