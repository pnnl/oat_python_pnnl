Glossary
========





.. glossary::









Matrices
------------------
.. glossary::


   Sparse matrix
      A **sparse matrix** is a matrix :math:`S` where each entry :math:`S_{ij}` has either an explicit numeric value, or a value of ``None``.

      By common convention, we call :math:`S_{ij}`
      
      - **structurally zero** if it is ``None``, and 
      - **structurally nonzero** if it has an explicit numeric value -- even if that value is zero.   

      **Contrast with other definitions**
      
      - The term `sparse matrix` has several other common uses; it often refers to any matrix with a large number of zeros.
      - Many authors treat ``None`` as zero entries which are not stored in computer memory. This assumption works
        poorly in the setting of topological data analysis. For example, if :math:`S` represents the adjacency matrix of a graph,
        then :math:`S_{ij} = None` typically indicate that there is no edge between vertices :math:`i` and :math:`j`,
        while :math:`S_{ij} = 0` typically indicates that there is an edge between :math:`i` and :math:`j` with weight zero.


   Scipy sparse CSR matrices
      A **Scipy sparse CSR matrix** is a sparse matrix represented in the Compressed Sparse Row (CSR) format.
      This format is efficient for arithmetic operations, matrix-vector products, and slicing.
      It is commonly used in Python for representing large sparse matrices, especially in scientific computing and data analysis.

      
      .. caution::
         **Unsorted indices**

         Scipy sparse CSR matrices do not require that the column indices of the
         nonzero entries be sorted (within each row).
         Moreover, there is no built-in method to check that this sorting
         condition is satisfied.
         Scipy CSR matrices do have a property flag called ``has_sorted_indices``,
         but this flag does not automatically update if the user manually modifies
         `matrix.indices`, which is sometimes necessary in topological data analysis.



   Matrix oracle
      A **matrix oracle** is an object that provides access to the entries of a matrix :math:`D` without storing the entire matrix in memory.
      It is typically used for large matrices where storing the entire matrix would be impractical.
      An oracle may provide methods to access individual entries, rows, or columns of the matrix as needed.

      **Example** In OAT, the :class:`oat_python.core.vietoris_rips.VietorisRipsMatrixOracle` class is an example of a matrix oracle.
      It provides access to the entries of a Vietoris-Rips boundary matrix without storing the entire matrix in memory.



   Umatch decomposition
      A **Umatch decomposition** of a matrix :math:`D` is a tuple of matrices :math:`(T, M, D, S)` such that

      - :math:`TM = DS`. This implicitly requres  :math:`S` and :math:`M`
        to have the same number of columns as :math:`D`, and :math:`T` and :math:`M`
        to have the same number of rows as :math:`D`.
      - :math:`S` and :math:`T` are square upper triangular matrices with ones on the diagonal
      - :math:`M` is a generalized matching matrix, meaning that :math:`M` has at most
        one nonzero entry per row and column


   Differential Umatch decomposition
      A **differential Umatch decomposition** for a simplicial complex :math:`K`
      is a :term:`Umatch decomposition` 
      for the :term:`differential matrix` :math:`D` of :math:`K`.
      We require this decomposition to have the form  :math:`(J, M, D, J)`,
      thus :math:`JM = DJ`. We also require
      `J` to be *homogeneous* with respect to dimension, meaning that each column of :math:`J` represents a linear
      combination of simplices of the same dimension.





Graphs and simplicial complexes
--------------------------------

.. glossary::

   Simple graph
      An undirected graph :math:`G = (V,E)` where each edge is an unordered pair of distinct vertices, i.e. :math:`E \subseteq \{\{i,j\} \mid i,j \in V, i \neq j\}`.

      Every simple graph is equivalent to an :term:`abstract simplicial complex` :math:`K` where 

      - The vertices of :math:`K` are the vertices of :math:`G`
      - :math:`K` contains a simplex :math:`\{v\}` for each vertex :math:`v \in V` and a simplex :math:`\{u,v\}` for each edge :math:`\{u,v\} \in E`.

   Abstract simplicial complex
      A set :math:`K` of finite sets (called simplices) such that if :math:`\sigma \in K` and :math:`\tau \subseteq \sigma`, then :math:`\tau \in K`.
      In other words, every subset of a simplex in :math:`K` is also a simplex in :math:`K`.
      - The elements of :math:`K` are called the simplices of the complex.
      - A simplex containing :math:`k+1` vertices is called a **k-simplex**.




   Vietoris Rips complex
      **Definition** Let :math:`G = (V,E)` be an unweighted simple graph. The **Vietoris Rips complex** of :math:`G` is
      the :term:`abstract simplicial complex` whose vertices are the vertices of :math:`G`, and whose simplices
      are the subsets of :math:`V` that induce a complete subgraph in :math:`G`. In other words, a subset
      :math:`\sigma \subseteq V` is a simplex in the Vietoris Rips complex if and only if every pair of vertices
      in :math:`\sigma` is connected by an edge in :math:`G`.

      **Filtrations and filter functions** Every :term:`filter function` :math:`w: E \cup V \to \mathbb{R}` defined on the vertices and edges of
      :math:`G` extends naturally to a filter function on the simplices of the associated Vietoris Rips complex :math:`K`.
      Specifically, for each simplex :math:`\sigma \in K`, the filtration value is defined as the maximum
      filtration value of its vertices and edges:

      .. math::

         w(\sigma) = \max \left ( \max_{v \in \sigma} w(v), \max_{e \in E(\sigma)} w(e)  \right )

      where :math:`E(\sigma)` is the set of edges contained in :math:`\sigma`.
      
      **Point clouds and metric spaces** Given a metric space or a point cloud, we can define the
      :term:`filtered <filter function>` Vietoris Rips complex as the filtered Vietoris Rips complex of the (filtered) complete graph
      :math:`G` where the vertices are the points in the point cloud,
      the filtration value of each vertex is zero, and the filtration value of each edge is the distance between the two points it connects.

      **(Sparse) dissimilarity matrices** The filtered Vietoris Rips complex of a (sparse) dissimilarity matrix :math:`D` is
      the filtered Vietoris Rips complex of the associated :term:`filtered simple graph <sparse dissimilarity matrix>`.


   Differential matrix
      The differential matrix of a simplicial complex :math:`K`
      is a matrix :math:`D` which has a row and column for each simplex in :math:`K`, and
      where :math:`\partial \sigma = \sum_{\tau \in K} D_{\tau, \sigma} \tau`
      for each simplex :math:`\sigma \in K`. This matrix is also called the **boundary matrix**
      of :math:`K`.


Filtrations
------------------

.. glossary::

   Filter function
      A **filter function** on an abstract simplicial complex :math:`K` is a function :math:`w: K \to \mathbb{R}`
      that assigns a real number to each simplex in :math:`K`.
      We require that the function be monotone, meaning that if :math:`\sigma \subseteq \tau` are simplices in :math:`K`, then
      
      .. math::

         w(\sigma) \leq w(\tau)

      In other words, the weight of a simplex cannot exceed the weight of any of its supersimplices.

      **Filtrations** The notion of a :term:`filter function` is interchangeable with that of a :term:`filtration`: the sublevel sets of a
      filter function :math:`w: K \to \mathbb{R}` define a filtration of :math:`K`, and conversely a filtration on :math:`K`
      defines a filter function by assigning to each simplex the minimum filtration parameter at which it appears.         

      **Simple graphs** A filter function on a simple graph :math:`G = (V,E)` is defined as a filter function :math:`w` on 
      the :term:`associated abstract simplicial complex<simple graph>`, :math:`K`.  This can be thought of, equivalently,
      as a function :math:`w': V \cup E \to \mathbb{R}` that assigns a real number to each vertex and edge of the graph, 
      where  :math:`w'(v) = w(\{v\})` for each vertex :math:`v`.

   Filtration
      A **filtration** of an abstract simplicial complex :math:`K` is a family of simplicial complexes
      :math:`(K_t)_{t \in \mathbb{R}}` such that for each :math:`t \in \mathbb{R}`, :math:`K_t` is a subcomplex of :math:`K`
      and if :math:`s \le t`, then :math:`K_s \subseteq K_t`. In other words, the filtration is a nested sequence of subcomplexes
      indexed by the real numbers.

      The notion of a :term:`filtration` is interchangeable with that of a :term:`filter function`: the sublevel sets of a
      filter function :math:`w: K \to \mathbb{R}` define a filtration of :math:`K`, and conversely a filtration on :math:`K`
      defines a filter function by assigning to each simplex the minimum filtration parameter at which it appears.

   Filtered simple graph
      This term refers to a :term:`simple graph` :math:`G = (V,E)` equipped with a :term:`filter function` :math:`w: V \cup E \to \mathbb{R}`.
      The vertices of the graph are assigned weights, and the edges are also assigned weights.           




Dissimilarity
------------------
.. glossary::

   Dissimilarity matrix
      This term refers to any (dense, as compared with sparse) symmetric :math:`n\times n` matrix :math:`D` where diagonal entries are the minima of their respective rows; that is,
      :math:`D_{ii} = \min_{j} D_{ij}` for all :math:`i`. 
      
      **Interpretation** This type of matrix is often used to represent dissimilarities between points in a point cloud,
      but it can also refer to matrices that represent 
      other measures of dissimilarity which don't satisfy the conditions of a metric space, e.g. subjective human judgements.

      **The associated filtered graph** is the complete graph :math:`G = (V,E)` on vertex set 
      :math:`1, \ldots, n`, equiped with the :term:`filter function` :math:`w: V \cup E \to \mathbb{R}` such that :math:`w(i) = D_{ii}` for all :math:`i`, and
      :math:`w(i,j) = D_{ij}` for all :math:`i \neq j`. In other words, the weight of each vertex is its diagonal entry in the matrix,
      and the weight of each edge is the corresponding off-diagonal entry. This weight function is an example of a :term:`filter function`, 
      because each vertex receives a value no greater than its incident edges.

   Sparse dissimilarity matrix
      This term refers to the sparse filtered adjacency matrix :math:`D` of :term:`filtered simple graph`. It can be characterized in several equivalent ways:

      **Definition 1** An :math:`n \times n` sparse matrix :math:`D` meets this condition if  for all :math:`i \neq j` in :math:`\{1, \ldots, n\}`,
      
      - :math:`D_{ij} = D_{ji}`
      - In particular :math:`D_{ij}` structurally nonzero if and only if :math:`D_{ji}` is structurally nonzero.
      - If row `i` contains one or more structural nonzero entries, then :math:`D_{ii}` is structurally nonzero, and :math:`D_{ii} = \min_j D_{ij}`
        where :math:`j` runs over all indices such that :math:`D_{ij}` is structurally nonzero.

      **Definition 2** An :math:`n \times n` sparse matrix :math:`D` meets this condition if there is a simple graph :math:`G = (V,E)`
      with vertex set :math:`V \subseteq \{1, \ldots, n\}` and edge set :math:`E \subseteq V \times V`, equipped
      with a :term:`filter function` :math:`w: V \cup E \to \mathbb{R}` such that for all :math:`i  \in \{1, \ldots, n\}`,

      .. math::      

         D_{ii} &= \begin{cases} w(\{i\}) & i \in V \\ \mathrm{None} & else \end{cases}

      and for all :math:`i \neq j` in :math:`\{1, \ldots, n\}`,
         
      .. math::

         D_{ij} &= \begin{cases} w(\{i,j\}) & \{i, j\} \in E \\ \mathrm{None} & else \end{cases}

      We call :math:`G` the **filtered simple graph** associated to :math:`D`.


      **Associated filtered graph** The associated filtered simple graph is the graph :math:`G = (V,E)` where 

      - The vertex set :math:`V` is the set of indices :math:`i` such that :math:`D_{ii}` is structurally nonzero.
      - The edge set :math:`E` is the set of pairs :math:`(i,j)` such that :math:`D_{ij}` is structurally nonzero, and
        the weight function :math:`w: V \cup E \to \mathbb{R}` is defined by

        - :math:`w(i) = D_{ii}` for all :math:`i \in V`
        - :math:`w(i,j) = D_{ij}` for all :math:`i \neq j` in :math:`V`.


   Dissimilarity space
      We use the term **dissimilarity space** interchangeablly with the term :term:`filtered simple graph`.
      In many cases, this graph is repersented by a :term:`sparse dissimilarity matrix`.

      In this context we often talk about the **dissimilarity** between two vertices :math:`i` and :math:`j`,
      meaning the filtration value (or weight) of the edge :math:`w(\{i,j\})`.




   Enclosing radius
   
      The **enclosing radius** of an :math:`n \times n` :term:`dissimilarity matrix` :math:`D_{ij}` is the minimum of the row-wise maxima of :math:`D_{ij}`.
      In symbols, it is defined as

      .. math::

         r_{\mathrm{enc}}(D) = \min_{i} \left( \max_{j} D_{ij} \right)
      
      **Empty matrices** If :math:`n = 0`, then we define :math:`r_{\mathrm{enc}}(D) = +\infty`.
      This is consistent with the convention that the minimum of an empty set is :math:`+\infty`.
      
      **Only for dense matrices** The enclosing radius is undefined for sparse dissimilarity matrices, although it can be used
      to sparsify a dense dissimilarity matrix. See below for details.
      
      **Interpretation for point clouds and metric spaces:**
      The enclosing radius can be interpreted in the context of point clouds and their dissimilarity matrices.
      If we have a point cloud :math:`X = \{x_1, x_2, \ldots, x_n\}` with a distance function :math:`d(\cdot, \cdot)`,
      then the enclosing radius captures the smallest maximum distance from any point to all other points in the cloud.
      
      **Significance in topological data analysis:**
      The enclosing radius marks an upper bound on the filtration values of a filtered Vietoris Rips complex that have interesting homology:

      .. admonition:: Theorem

         Suppose that :math:`n \ge 0`, and let :math:`(K_t)_{t \in \mathbb{R}}` be the filtered :term:`Vietoris Rips complex <vietoris rips complex>` of :math:`D`.
         Then the homology of :math:`K_t` is trivial (formally, isomorphic to the homology of a single point) for all :math:`t \geq r_{\mathrm{enc}}(D)`.

         Consequently, if :math:`H` is the sparse dissimilarity matrix obtained from :math:`D` by deleting
         entry :math:`D_{ij}` for all :math:`i \neq j` such that :math:`D_{ij} > r_{\mathrm{enc}}(D)`, then the filtered Vietoris Rips complexes of :math:`D` and
         :math:`H` have isomorphic persistent homology.
      
      This has significant practical implications, because the time and memory cost of computing persistent homology of the filtered Vietoris Rips complex
      of :math:`H` is often orders of magnitude smaller than that of :math:`D`.


      .. caution::

         **Numerical error**
      
         Distance computations in Python are subject to numerical error.
         Therefore, if you use the enclosing radius to sparsify a dissimilarity matrix for a Vietoris-Rips persistent homology calculation, it's
         important to add a small buffer, e.g. ``enclosing_radius + 0.00000001``, to avoid over-sparsification.
         
         **Details**

         Numerical error appears in many places in Python; for example,

         - ``euclidean_distance_one_sided( pointa, pointb )`` sometimes returns a value
           slightly different from ``euclidean_distance_one_sided( pointb, pointa )``, even though the only
           difference between the two calls is the order of the arguments.
         - ``sklearn.neighbors.radius_neighbors_graph`` often produces asymmetric matrices
         - ``sklearn.metrics.pairwise_distances`` often produces asymmetric matrices

         Moreover, it's common to calculate the same quantity multiply times in different ways for a single workflow, resulting in
         slightly different error values.
         Suppose, for example, that we want to compute the Vietoris Rips persistent homology of a point cloud that has many points.
         In this case, we often want to avoid computing a dense distance matrix, because it is expensive to compute and store.
         Instead, we want to directly compute a :term:`sparse dissimilarity matrix` where all values greater than the enclosing radius are removed.
         To do this, we can 
      
         - First calculate the enclosing radius of the distance matrix by calling ``oat_python.dissimilarity.enclosing_radius_for_cloud(cloud)``.
           This function only holds one row of the distance matrix in memory at a time,
           so it is much more memory-efficient than computing the entire distance matrix.
         - Then calculate the :term:`sparse dissimilarity matrix` by calling ``oat_python.dissimilarity.sparse_dissimilarity_matrix_for_cloud(cloud, enclosing_radius + 0.00000001)``.

         The reason for adding ``0.00000001`` is that distances computed by the function ``oat_python.dissimilarity.enclosing_radius_for_cloud`` may vary slightly
         from those computed by ``oat_python.dissimilarity.sparse_dissimilarity_matrix_for_cloud``. Therefore, if we are
         unlucky, then calling ``oat_python.dissimilarity.sparse_dissimilarity_matrix_for_cloud(cloud, enclosing_radius)`` may delete some
         entries that should not be deleted.
         
         In summary
         
         - Adding a buffer value of `0.00000001`` (or some other small value) to the enclosing radius ensures that we do not delete important entries.
         - This may result in a slightly larger sparse dissimilarity matrix than necessary, but the persistent homology calculation
           on this larger matrix will be correct, and the additional time and memory needed to compute persistent homology will be negligible.
         - No buffer is needed if you know that all distances used for the enclosing radius and the sparse dissimilarity matrix
           are computed in exactly the same way. For example, ``oat_python.dissimilarity.enclosing_radius_for_cloud_slow`` and
           ``oat_python.dissimilarity.sparse_dissimilarity_matrix_for_cloud_slow`` are specifically engineered to work together,
           with no need for a buffer.

      