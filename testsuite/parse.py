# Copyright (c) 2018, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-745557. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the ParElag library. For more information and source code
# availability see http://github.com/LLNL/parelag.
#
# ParElag is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

"""
read output from test routines (notably UpscalingGeneralForm.exe)
and parse it, returning a dictionary of errors, dofs, etc.

this used to be part of test.py, taken out for modularity
"""

import sys

def linehas(p,num,val):
    if len(p) > num and p[num] == val:
        return True
    else:
        return False

def parse_output(lines,derivative="der"):
    """
    for now we are only looking at the first column of the error 
    output, looking at upscaling error convergence rates with respect
    to the finest level
    """
    out = {}
    out["l2error"] = []
    out["derror"] = []
    out["perror"] = []
    out["dofs"] = []
    out["nnzs"] = []
    out["iterations"] = []
    out["assembly-times"] = []
    out["precondition-times"] = []
    out["solver-times"] = []
    mode = "none"
    for line in lines: 
        p = line.split()
        if linehas(p,0,"%level") or linehas(p,0,"%ulevel"):
            if linehas(p,4,"EMPTY"):
                mode = "doftable"
            else:
                mode = "unstructured-doftable"
        elif linehas(p,0,"%") and linehas(p,2,"uh") and linehas(p,4,"uH"):
            mode = "l2error"
            continue
        elif linehas(p,0,"%") and linehas(p,2,"uh") and linehas(p,3,"||"):
            mode = "l2norm"
            continue
        elif linehas(p,0,"%") and linehas(p,2,"ph") and linehas(p,4,"pH"):
            mode = "perror"
            continue
        elif linehas(p,0,"%") and linehas(p,2,"ph") and linehas(p,3,"||"):
            mode = "pnorm"
            continue
        elif linehas(p,0,"%") and linehas(p,2,derivative) and linehas(p,3,"("):
            mode = "derror"
            continue
        elif linehas(p,0,"%") and linehas(p,2,derivative) and linehas(p,3,"uh"):
            mode = "dnorm"
            continue
        elif linehas(p,0,"}") and len(p) == 1:
            mode = "none"
        else:
            if mode == "doftable":
                out["dofs"].append(int(p[1]))
                out["nnzs"].append(int(p[2]))
                out["iterations"].append(int(p[4]))
                out["assembly-times"].append(float(p[5]))
                out["precondition-times"].append(float(p[7]))
                out["solver-times"].append(float(p[9]))
            elif mode == "unstructured-doftable":
                out["dofs"].append(int(p[1]))
                out["nnzs"].append(int(p[2]))
                out["iterations"].append(int(p[3]))
                out["assembly-times"].append(float(p[4]))
                out["precondition-times"].append(float(p[5]))
                out["solver-times"].append(float(p[6]))                
            elif mode == "l2error": 
                out["l2error"].append(float(p[0]))
            elif mode == "l2norm":
                out["l2norm"] = float(p[0])
            elif mode == "derror":
                out["derror"].append(float(p[0]))
            elif mode == "dnorm":
                out["dnorm"] = float(p[0])
            elif mode == "perror":
                out["perror"].append(float(p[0]))
            elif mode == "pnorm":
                out["norm"] = float(p[0])
    return out

if __name__ == "__main__":
    fd = open(sys.argv[1],"r")
    print parse_output(fd,sys.argv[2])
    fd.close()
