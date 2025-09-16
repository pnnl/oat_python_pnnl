"""
.. _dowker_relabel_reduce_gallery:

Reduce & Relabel
=========================================

Homology calculations are only available in OAT for one type of hypergraph: a list of sorted-lists of integers.
**However** we provide tools to translate back and forth between this and other data fromats, e.g. a dictionary of lists of strings.  

Below are two examples.  
"""


import oat_python as oat

# %%
# See also
# -----------------------------------------------
# 
# - The `HypernetX <https://pnnl.github.io/HyperNetX/>`_ library provides many useful tools for working with hypergraphs,
#   including tools to remove duplicate nodes and edges.
# - :ref:`dowker_rbs_homology_gallery` shows how to relabel nodes/edges, and use proper labels in plotting.


# %%
# Relabel a hypergraph
# -----------------------------------------------
#
# Here we assign integer labels to the nodes of a hypergraph whose nodes are labeled with strings.

# %%
# Define a hypergraph
E = { "A": ["x"], "B": ["y"], "C": ["x","y","z","zz"], "D": ["w","ww","x","y"], "DD": ["w","ww","x","y"] }

# %%
# Relabel with integers
relabeled_hg, label_translator = oat.hypergraph.relabel(E)

# %%
# The relabeled hypergraph:
relabeled_hg

# %%
# The label translator maps the new integer labels back to original labels and vice versa
for key, val in label_translator.items():
    print(f"{key}:  {val}")

# %%
# Relabel **and reduce** a hypergraph
# -----------------------------------------------
# 
# Here we perform two actions in one step:
# 
# - reduce the hypergraph by removing duplicate hyperedges (meaning hyperedges with the same vertex set) and duplicate nodes (meaning nodes that belong to the same set of hyperedges)
# - reformat the reduced hypergraph as a list of sorted-lists of integers

# %%
# Define a hypergraph
E = { "A": ["x"], "B": ["y"], "C": ["x","y","z","zz"], "D": ["w","ww","x","y"], "DD": ["w","ww","x","y"] }

# %%
# Collapse out redundant information, returning a list of lists
reduced_hg, label_translator = oat.hypergraph.reduce_hypergraph_with_labels(E)

# %%
# The reduced hypergraph edges (duplicate nodes have been removed; the remaining nodes have been relabeled as integers)
reduced_hg

# %%
# The label translator maps 
#
# - new integer labels back to **sets of original labels**
# - **single original labels** back to new integer labels
for key, val in label_translator.items():
    print(f"{key}:  {val}")