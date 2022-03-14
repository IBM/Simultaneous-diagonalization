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
import qiskit
import cl


class RecordCircuit(cl.Recorder) :
   def __init__(self, nQubits) :
      super(RecordCircuit,self).__init__()
      self.nQubits = nQubits
      self.reset()
      self.singleRz = False

   def reset(self) :
      self.circuit = qiskit.QuantumCircuit(self.nQubits)

   def addH(self,index) :
      self.circuit.h(index)

   def addS(self,index) :
      self.circuit.s(index)

   def addSdg(self,index) :
      self.circuit.sdg(index)

   def addX(self,index) :
      self.circuit.x(index)

   def addY(self,index) :
      self.circuit.y(index)

   def addZ(self,index) :
      self.circuit.z(index)

   def addRz(self,index,angle) :
      if (self.singleRz) :
         # Assume Rz = Diag(exp(i*angle),exp(-i*angle))
         self.circuit.rz(angle,index)
      else :
         self.circuit.rz(-angle,index)
         self.circuit.x(index)
         self.circuit.rz(angle,index)
         self.circuit.x(index)

   def addCNot(self,index1,index2) :
      self.circuit.cx(index1,index2)


class RecordCircuitCode(cl.Recorder) :
   def __init__(self, nQubits) :
      super(RecordCircuitCode,self).__init__()
      self.nQubits = nQubits
      self.reset()

   def reset(self) :
      self.code = []
      self.code.append('circuit = qiskit.QuantumCircuit(%d)' % self.nQubits)

   def addH(self,index) :
      self.code.append('circuit.h(%d)' % index)

   def addS(self,index) :
      self.code.append('circuit.s(%d)' % index)

   def addSdg(self,index) :
      self.code.append('circuit.sdg(%d)' % index)

   def addX(self,index) :
      self.code.append('circuit.x(%d)' % index)

   def addY(self,index) :
      self.code.append('circuit.y(%d)' % index)

   def addZ(self,index) :
      self.code.append('circuit.z(%d)' % index)

   def addCNot(self,index1,index2) :
      self.code.append('circuit.cx(%d,%d)' % (index1,index2))
