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
Parse output from valgrind

originally copied from saamge_parallel/proj/amg/test/mltest/test.py
"""

def valgrindparse(lines):
    out = {}
    for line in lines:
        p = line.split()
        if len(p) > 5 and p[1] == "definitely" and p[2] == "lost:":
            out["definitely-lost"] = int(p[3].replace(',',''))
        if len(p) > 5 and p[1] == "indirectly" and p[2] == "lost:":
            out["indirectly-lost"] = int(p[3].replace(',',''))
        if len(p) > 5 and p[1] == "possibly" and p[2] == "lost:":
            out["possibly-lost"] = int(p[3].replace(',',''))
        if len(p) > 5 and p[1] == "still" and p[2] == "reachable:":
            out["still-reachable"] = int(p[3].replace(',',''))
        if len(p) > 10 and p[1] == "ERROR" and p[2] == "SUMMARY:":
            out["total-errors"] = int(p[3])
        if len(p) > 5 and p[1] == "All" and p[2] == "heap" and p[5] == "freed":
            out["definitely-lost"] = 0
            out["indirectly-lost"] = 0
            out["possibly-lost"] = 0
            out["still-reachable"] = 0
    return out

if __name__ == "__main__":
    fd = open(sys.argv[1],"r")
    lines = fd.readlines()
    fd.close()
    print(valgrindparse(lines))
