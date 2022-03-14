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

import os
import json
import networkx as nx
import cl
import numpy as np
import functools

from pathlib import Path


hamiltonian_dir = "Hamiltonians"
prefix_coloring = "partition_"
postfix_paulis  = "_Pauli_list.txt"
postfix_qubits  = "qubits"


def load_hamiltonian(filename) :
   with open(filename,'r') as fp :
      data = fp.read()

   if (data[0] == '{') :
      # Assume JSON
      data = json.loads(data)
      data = data["paulis"]
      paulis = []
      weights = []
      for element in data :
          paulis.append(element["label"].lower())
          weights.append(float(element["coeff"]["real"]))
   else :
      data = data.splitlines()
      paulis  = [p.lower() for p in data[0::2]]
      weights = [complex(d).real for d in data[1::2]]

   return [x for x in zip(paulis,weights)]


def symbol_latex(symbol) :
   symbol_tex = ''
   flagNumeric = False
   for c in symbol+' ' :
      if (c.isdigit()) :
         if (not flagNumeric) :
            symbol_tex += '$_{'
         flagNumeric = True
      else :
         if (flagNumeric) :
            symbol_tex += '}$'
         flagNumeric = False
      symbol_tex += c
   symbol_tex = symbol_tex[:-1]
   return symbol_tex


def index_molecule(path) :
   pattern = postfix_paulis
   entries = []
   for entry in Path(path).glob("*"+pattern) :
      molecule = Molecule(entry)
      entries.append((molecule.encoding, molecule))

   entries.sort()
   return entries

def index_molecules() :
   entries = []
   for entry in Path(hamiltonian_dir).glob("*" + postfix_qubits) :
      instances = index_molecule(entry)
      if (len(instances) > 0) :
         entries.append((instances[0][1].basename, instances))
   return entries


def partition_sequential(paulis) :
   partition = []
   for i in range(len(paulis)) :
      found = False
      for j in range(len(partition)) :
         commutes = True
         for index in partition[j] :
            if (not cl.paulis_commute(paulis[index][0],paulis[i][0])) :
               commutes = False
               break

         if commutes :
            partition[j].append(i)
            found = True
            break

      if (not found) :
         partition.append([i])

   return partition


def partition_hamiltonian(paulis, strategy='largest_first') :
   if (strategy == 'sequential') :
     return partition_sequential(paulis)

   g = nx.Graph() # Undirected graph
   g.add_nodes_from(range(len(paulis))) # Make sure all paulis are added
   for i in range(len(paulis)) :
      for j in range(i+1,len(paulis)) :
         if (not cl.paulis_commute(paulis[i][0],paulis[j][0])) :
            g.add_edge(i,j)

   c = nx.greedy_color(g, strategy=strategy)
   d = {}
   for (k,v) in c.items() :
      l = d.get(v,[])
      l.append(k)
      d[v] = l

   return [value for (key,value) in d.items()]


class Molecule(object) :
   def __init__(self,fullname) :
      self.fullname = Path(fullname)
      self.folder   = self.fullname.parts[-2]
      self.filename = self.fullname.parts[-1]

      # Extract folder information
      s = self.folder.split("_")
      self.basename       = self.folder[:-len(postfix_qubits)]
      self.molecule       = s[0]
      self.molecule_latex = symbol_latex(self.molecule)
      self.basis          = s[1]
      self.basis_star     = s[1].replace('*','star')
      self.nQubits        = int(s[2][:-len(postfix_qubits)])

      # Extract filename information
      name = self.filename; offset = 0
      for i in range(len(name)) :
         ch = name[-(i+1)]
         if ((ch not in ['_','.']) and (not ch.isalpha())) :
            offset = len(name) - i
            break

      self.file_prefix = name[:offset]
      self.encoding = name[offset:-len(postfix_paulis)]
      self.encoding_mixed = self.encoding.replace("_"," ").title()
      self.encoding_abbrv = "".join([s[0] for s in self.encoding_mixed.split()])
      self.encoding_bar   = self.encoding_mixed.replace(" ","-")

      self.paulis = None

   def Hamiltonian(self) :
      if (not self.paulis) :
         self.paulis = load_hamiltonian(self.fullname)
      return self.paulis

   def coloring_filename(self, strategy) :
      filename = "%s%s_Coloring_%s.txt" % (self.file_prefix, self.encoding, strategy)
      return filename

   def coloring_fullname(self, path, strategy) :
      if (path is None) :
         path = hamiltonian_dir
      filename = self.coloring_filename(strategy)
      return Path(path,self.folder,filename)

   def coloring(self, strategy="sequential", cache=True, compute=True, rerun=False, path=None) :
      # Set the filename for caching
      fullname = self.coloring_fullname(path, strategy)

      # See if the result is cached
      if ((cache) and (not rerun)) :
         try :
            with open(fullname,"r") as fp :
               partition = []
               for line in fp.readlines() :
                  partition.append([int(element) for element in line.split()])
            return partition
         except FileNotFoundError:
            pass

      # Compute the result if needed
      if (not compute) :
         return None

      # Get the Pauli terms and generate the partition
      paulis = self.Hamiltonian()
      partition = partition_hamiltonian(paulis, strategy)

      # Validate the result
      if ((sum([len(p) for p in partition]) != len(paulis)) or
          (len(set(functools.reduce((lambda x,y : x+y), partition))) != len(paulis))) :
         # The number of elements must match and there should be no duplicates
         raise Exception("Invalid coloring")

      # Store the result if needed
      if (cache) :
         if (path is None) :
            path = hamiltonian_dir
         cl.ensureDirExists(path)

         # Write the file
         with open(fullname,"w") as fp :
             for group in partition :
                fp.write(" ".join([str(idx) for idx in group]))
                fp.write("\n")

      # Return the result
      return partition
