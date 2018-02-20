
### Element-Agglomeration Algebraic Multigrid and Upscaling Library

version 2.0

``` 
      ________              _______________________________
      ___  __ \_____ __________  ____/__  /___    |_  ____/
      __  /_/ /  __ `/_  ___/_  __/  __  / __  /| |  / __  
      _  ____// /_/ /_  /   _  /___  _  /___  ___ / /_/ /  
      /_/     \__,_/ /_/    /_____/  /_____/_/  |_\____/   
                                                                 
```

https://github.com/llnl/parelag

ParElag is a parallel distributed memory C++ library for numerical
upscaling of finite element discretizations.

For building instructions, see the file INSTALL. Copyright information
and licensing restrictions can be found in the file COPYRIGHT.

ParElag implements upscaling and algebraic multigrid techniques for
the efficient solution of the algebraic linear system arising from
mixed finite element discretization of saddle point problems.  In
particular, it constructs upscaled discretization for wide classes of
partial differential equations and unstructured meshes in an
element-based algebraic multigrid framework. This approach has the
potential to be more accurate than classical upscaling techniques
based on piecewise polynomial approximation. In fact, the coarse
spaces and respective coarse models computed by ParElag not only
posses the same stability and approximation properties of the original
finite element discretization but also are operator dependent.

ParElag uses the MFEM library (http://mfem.org) for the finite element
discretization and supports several solvers from the HYPRE library.
Visualization in ParElag is based on GLvis (http://glvis.org).

For examples of using ParElag, see the examples/ directory.

This work was performed under the auspices of the U.S. Department of
Energy by Lawrence Livermore National Laboratory under Contract
DE-AC52-07NA27344. LLNL-CODE-745557.
