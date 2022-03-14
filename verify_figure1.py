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

import qiskit
import numpy as np

c = qiskit.QuantumCircuit(4)
c.h(1)
c.h(2)
c.cx(1,3)
c.cx(2,3)
c.rz(0.2,3) # Theta1
c.cx(2,3)
c.cx(1,3)
c.h(1)
c.h(2)

c.s(1)
c.h(1)
c.x(1)
c.cx(0,3)
c.cx(1,3)
c.cx(2,3)
c.rz(0.4,3) # Theta2
c.cx(2,3)
c.cx(1,3)
c.cx(0,3)
c.x(1)
c.h(1)
c.sdg(1)

c.h(0)
c.h(1)
c.cx(0,3)
c.cx(1,3)
c.rz(0.34,3) # Theta3
c.cx(1,3)
c.cx(0,3)
c.h(0)
c.h(1)

cTop = c
print(c.draw())


c = qiskit.QuantumCircuit(4)
c.h(1)
c.h(2)
c.cx(1,3)
c.cx(2,3)
c.rz(0.2,3) # Theta1
c.cx(2,3)
c.cx(1,3)
c.h(1)
c.h(2)

c.h(0)
c.h(1)
c.cx(0,3)
c.cx(1,3)
c.rz(0.34,3) # Theta3
c.cx(1,3)
c.cx(0,3)
c.h(0)
c.h(1)

c.s(1)
c.h(1)
c.x(1)
c.cx(0,3)
c.cx(1,3)
c.cx(2,3)
c.rz(0.4,3) # Theta2
c.cx(2,3)
c.cx(1,3)
c.cx(0,3)
c.x(1)
c.h(1)
c.sdg(1)

cBottom = c
print(c.draw())

cOptimized = qiskit.compiler.transpile(c, optimization_level=2)
print(cOptimized.draw())


backend = qiskit.Aer.get_backend('unitary_simulator')
U0 = qiskit.execute(cTop, backend).result().get_unitary()
U1 = qiskit.execute(cBottom, backend).result().get_unitary()
U2 = qiskit.execute(cOptimized, backend).result().get_unitary()

print("Difference ||U0 - U1|| = %s" % (np.linalg.norm(U0-U1,'fro')))
print("Difference ||U0 - U2|| = %s" % (np.linalg.norm(U0-U2,'fro')))
print("Difference ||U1 - U1|| = %s" % (np.linalg.norm(U1-U2,'fro')))
