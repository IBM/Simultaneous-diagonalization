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
import cl_qiskit as clq
import cl_methods as clm
import cl_chemistry as clc
import numpy as np
import math


# Get list of molecules
molecules = clc.index_molecules()
molecules.sort()

# Generation of filename for caching
def getFilename(instance, strategy) :
   return "chemistry_%s_%s_%s_%s" % (instance.molecule,
                                     instance.basis_star,
                                     instance.encoding_abbrv.lower(),
                                     strategy)


# ======================================================================
# Methods for benchmarking
# ======================================================================

def _apply_diag_info(T,R, weights) :
   # Circuit with optimization
   RC = clq.RecordCircuit(T.n+1); RC.singleRz = True
   clm.circuitDiagExponentiateGenerate(RC, R, T, weights, funOrderZ=cl.orderZ)
   C  = RC.circuit

   # Gather information
   c = C.count_ops()
   return (c.get('cx',0),sum([value for (key,value) in c.items()])-c.get('cx',0), C.depth())


def _apply_method_cnot(T, weights, blocksizes) :
   R = cl.RecordOperations(T.n)
   blocksize = clm.circuitDiagonalize_CNot(T,R,blocksizes)
   return _apply_diag_info(T,R, weights)


def _apply_method_generic(M, weights, method) :
   T = cl.Tableau(M)
   R = cl.RecordOperations(T.n)
   clm.circuitDiagonalize(T,R,method)
   return _apply_diag_info(T,R,weights)


# User callable methods - simultaneous diagonalization
def apply_method_cnot(M, weights) :
   T = cl.Tableau(M)
   return _apply_method_cnot(T, weights, [])

def apply_method_cnot_log2(M, weights) :
   T = cl.Tableau(M)
   return _apply_method_cnot(T, weights, [int(math.ceil(math.log2(T.n)))])

def apply_method_cnot_best(M, weights) :
   T = cl.Tableau(M)
   return _apply_method_cnot(T, weights, range(1,T.n+1))

def apply_method_cz(M, weights) :
   return _apply_method_generic(M, weights, cl.zeroX_algorithm1_cz)

def apply_method_greedy0(M, weights) :
   return _apply_method_generic(M, weights, cl.zeroX_algorithm1_greedy0)

def apply_method_greedy1(M, weights) :
   return _apply_method_generic(M, weights, cl.zeroX_algorithm1_greedy1)

def apply_method_greedy2(M, weights) :
   return _apply_method_generic(M, weights, cl.zeroX_algorithm1_greedy2)

def _apply_diag_info_rnd(T,R, weights,trials) :
   # Circuit with optimization
   RC = clq.RecordCircuit(T.n+1); RC.singleRz = True
   funOrderZ = (lambda x : cl.orderZ_randomized(x,trials))
   clm.circuitDiagExponentiateGenerate(RC, R, T, weights, funOrderZ=funOrderZ)
   C  = RC.circuit

   # Gather information
   c = C.count_ops()
   return (c.get('cx',0),sum([value for (key,value) in c.items()])-c.get('cx',0), C.depth())

def _apply_method_generic_rnd(M, weights, method, trials) :
   T = cl.Tableau(M)
   R = cl.RecordOperations(T.n)
   clm.circuitDiagonalize(T,R,method)
   return _apply_diag_info_rnd(T,R,weights,trials)

def apply_method_cz_rnd(M, weights) :
   return _apply_method_generic_rnd(M, weights, cl.zeroX_algorithm1_cz, 100)


# User callable methods - direct exponentiation
def apply_method_direct(M, weights) :
   index = cl.orderPaulis(M)
   Mbar = M[index,:]
   T = cl.Tableau(Mbar)
   R = cl.RecordOperations(T.n+1)
   clm.circuitDirectExponentiation(R, cl.matrix_to_pauli(Mbar), weights, optimized=True)

   # Circuits
   RC = clq.RecordCircuit(R.nQubits); RC.singleRz = True
   R.sendTo(RC)
   C = clq.qiskit.compiler.transpile(RC.circuit, optimization_level=2)

   # Gather information
   c = C.count_ops()
   return (c.get('cx',0),sum([value for (key,value) in c.items()])-c.get('cx',0), C.depth())


def apply_method_direct_rnd(M, weights) :
   index = cl.orderPaulis_randomized(M,100)
   Mbar = M[index,:]
   T = cl.Tableau(Mbar)
   R = cl.RecordOperations(T.n+1)
   clm.circuitDirectExponentiation(R, cl.matrix_to_pauli(Mbar), weights, optimized=True)

   # Circuits
   RC = clq.RecordCircuit(R.nQubits); RC.singleRz = True
   R.sendTo(RC)
   C = clq.qiskit.compiler.transpile(RC.circuit, optimization_level=2)

   # Gather information
   c = C.count_ops()
   return (c.get('cx',0),sum([value for (key,value) in c.items()])-c.get('cx',0), C.depth())



# ======================================================================
# Experiment setup
# ======================================================================

methods = [("cz",           apply_method_cz,            "cz"),
           ("rnd_cz",       apply_method_cz_rnd,        "cz-rnd"),
           ("cnot",         apply_method_cnot,          "cnot"),
           ("cnot_log2",    apply_method_cnot_log2,     "cnot-log2"),
           ("cnot_best",    apply_method_cnot_best,     "cnot-best"),
           ("greedy1",      apply_method_greedy1,       "greedy-1"),
           ("greedy2",      apply_method_greedy2,       "greedy-2"),
           ("original",     apply_method_direct,        "direct"),
           ("rnd_original", apply_method_direct_rnd,    "direct-rnd")]

methods2 = [methods[0],methods[5],methods[6],methods[7]]


# ======================================================================
# Generate all required results
# ======================================================================

def getInstance(molecule, encoding) :
   # Get the instance corresponding to the given encoding
   instance = [inst[1] for inst in molecule[1] if (inst[1].encoding_abbrv.lower() == encoding)]
   if (len(instance) > 1) :
      raise Exception("Encoding '%s' appears more than once" % str(encoding))
   return instance[0] if instance else None


def getResult(instance, strategy, method, compute=True) :
   if (instance is None) :
       return None

   # Get the coloring
   coloring = instance.coloring(strategy=strategy, cache=True, compute=False)
   if (coloring is None) :
      print("Warning: Graph coloring is disabled; please run the generate_coloring.py script first")
      return None

   # Extract the method components
   (method_infix, method_fun, method_name) = method


   # Check if we need to combine different methods
   if (isinstance(method_fun, tuple)) :
      for (idx,method) in enumerate(method_fun) :
         data = getResult(instance, strategy, method, compute)
         if (idx == 0) :
            output = data
         else :
            # Elementwise minimum
            output = [(min(v1[0],v2[0]),min(v1[1],v2[1]),min(v1[2],v2[2])) for (v1,v2) in zip(data,output)]
      return output

   # Get the result cache
   filename = getFilename(instance, strategy)
   cache = cl.ResultCache(filename)

   output = []; hamiltonian = None
   for (i,indices) in enumerate(coloring) :
        key = (method_infix, i)
        if (key in cache) :
           output.append(cache[key])
           continue
        elif (not compute) :
           return None

        # Get the Hamiltonian
        if (hamiltonian is None) :
           hamiltonian = instance.Hamiltonian()


        print("%s %d/%d (%d) -- %s" % (filename,i, len(coloring),len(indices),method_name))
        paulis  = [hamiltonian[idx][0] for idx in indices]
        weights = [hamiltonian[idx][1] for idx in indices]
        weights = np.asarray(weights)
        M       = cl.pauli_to_matrix(paulis)
        result  = method_fun(M,weights)

        output.append(result)
        cache[key] = result

   # Make sure cache upates are flushed to disk
   cache.flush()

   return output



def generateTable(methods, encoding, strategy, filename=None) :
   instances = [getInstance(molecule, encoding) for molecule in molecules]
   instances = [instance for instance in instances if (instance is not None)]

   # Generate the header
   header  = "\\begin{tabular}{|l|%s|}\n" % ("r" * len(instances))
   header += "\\hline\n"

   header += "{\\bf{Method}}"
   for instance in instances :
      header += "& %s" % (instance.molecule_latex)
   header += "\\\\[-5pt]\n"
   header += ""
   for instance in instances :
      header += "& {\\tiny{%s}}" % (instance.basis)
   header += "\\\\\n"
   header += "\\hline\n"

   # Generate the table body
   body = ["","",""]

   # Generate the entries (cnot, single, depth)
   block  = np.zeros((len(methods),len(instances),3),np.double)
   for [i,method] in enumerate(methods) :
      for [j,instance] in enumerate(instances) :
         result = getResult(instance, strategy, method, compute=False)
         if (result) :
            for k in [0,1,2] :
               block[i,j,k]   = sum([r[k] for r in result])
         else :
            block[i,j,:] = np.nan

   # Get the optimal values
   highlight = (block == np.nanmin(block,axis=0,keepdims=True))

   for [i,method] in enumerate(methods) :
      # Extract the method components
      (method_infix, method_fun, method_name) = method
      for k in [0,1,2] :
         body[k] = body[k] + method_name

      for (j,instance) in enumerate(instances) :
         for k in [0,1,2] :
            if (not np.isnan(block[i,j,k])) :
               v = format(int(block[i,j,k]),",")
               if (highlight[i,j,k]) :
                  v = "\\cellcolor{highlight}{\\color{darkblue}%s}" % v
                  #v = "{\\bf{%s}}" % v
            else :
               v = "--"
            body[k] = body[k] + ("&" + v)

      for k in [0,1,2] :
         body[k] = body[k] + "\\\\\n"

   # Generate the table footer
   footer = "\\end{tabular}"

   # Table
   table  = header
   table += body[0] + "\\hline\n"
   table += "\\multicolumn{%d}{c}{\\cnot\\ count}\\\\[6pt]\n\\hline\n" % (len(instances)+1)
   table += body[1] + "\\hline\n"
   table += "\\multicolumn{%d}{c}{Single-qubit count}\\\\[6pt]\n\\hline\n" % (len(instances)+1)
   table += body[2] + "\\hline\n"
   table += "\\multicolumn{%d}{c}{Circuit depth}\n" % (len(instances)+1)
   table += footer

   if (filename) :
      with open(filename,"w") as fp :
         fp.write(table)
   else :
      return (len(instances), header, body[0], body[1], body[2], footer)


# Make sure the tables directory exists
cl.ensureDirExists('tables')

# Generate the first table
(na,ha,ba1,ba2,ba3,fa) = generateTable(methods, 'jw', 'sequential')
(nb,hb,bb1,bb2,bb3,fb) = generateTable(methods, 'bk', 'sequential')
(nb,hb,bc1,bc2,bc3,fc) = generateTable(methods, 'p',  'sequential')
filename = "tables/table_simulate_sequential.tex"
print("Writing %s . . ." % filename)
with open(filename,"w") as fp :
    fp.write(ha)
    fp.write(ba1)
    fp.write("\\hline\n")
    fp.write("\\multicolumn{%d}{c}{\\cnot\\ count}\\\\[8pt]\n" % (na+1))
    fp.write("\\hline\n")
    fp.write(ba3)
    fp.write("\\hline\n")
    fp.write("\\multicolumn{%d}{c}{Circuit depth}\\\\[8pt]\n" % (na+1))
    fp.write("\\hline\n")
    fp.write(ba2)
    fp.write("\\hline\n")
    fp.write(bb2)
    fp.write("\\hline\n")
    fp.write(bc2)
    fp.write("\\hline\n")
    fp.write("\\multicolumn{%d}{c}{Single-qubit count}\n" % (na+1))
    fp.write(fa)


# Perpare for the second and third tables
info = []
strategies = ['sequential','largest_first','independent_set']
for strategy in strategies :
   for encoding in ['jw','bk','p'] :
      info.append(generateTable(methods2, encoding, strategy))


# Generate the second table
filename = "tables/table_simulate_other_cnot.tex"
print("Writing %s . . ." % filename)
with open(filename,"w") as fp :
    fp.write(info[0][1])
    for (idx,d) in enumerate(info) :
       fp.write(d[2])
       fp.write("\\hline\n")
       if (idx % 3 == 2) :
          strategy = strategies[idx//3].replace("_"," ")
          fp.write("\\multicolumn{%d}{c}{%s}" % (d[0]+1,strategy))
          if (idx < 8) :
             fp.write("\\\\[6pt]\n\\hline\n")
          else :
             fp.write("\n")
    fp.write(info[0][-1])


# Generate the third table
filename = "tables/table_simulate_other_depth.tex"
print("Writing %s . . ." % filename)
with open(filename,"w") as fp :
    fp.write(info[0][1])
    for (idx,d) in enumerate(info) :
       fp.write(d[4])
       fp.write("\\hline\n")
       if (idx % 3 == 2) :
          strategy = strategies[idx//3].replace("_"," ")
          fp.write("\\multicolumn{%d}{c}{%s}" % (d[0]+1,strategy))
          if (idx < 8) :
             fp.write("\\\\[6pt]\n\\hline\n")
          else :
             fp.write("\n")
    fp.write(info[0][-1])
