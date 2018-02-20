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

from __future__ import print_function

import commands
import getpass
import os
import platform
import tempfile

class Job(object):
    def __init__(self,name="job",np=1,nodes=None):
        self.name = name
        if "aztec" in platform.node():
            self.partition = "aztec"
        elif "cab" in platform.node():
            self.partition = "cab"
        elif "sierra" in platform.node():
            self.partition = "sierra"
        else:
            raise ValueError
        self.np = np
        self.nodes = nodes
        self.workingdir = "/p/lscratchd/" + getpass.getuser() + "/testoutput/"
    def getscript(self,commandline):
        out = "#!/bin/bash\n\n"
        if self.nodes == None:
            out = out + "#MSUB -l procs=" + str(self.np) + "\n"
        else:
            out = out + "#MSUB -l nodes=" + str(self.nodes) + "\n"
        out = out + "#MSUB -l partition=" + self.partition + "\n"
        out = out + "#MSUB -l walltime=4:00:00\n"
        out = out + "#MSUB -q pbatch\n"
        out = out + "#MSUB -N " + self.name + "\n"
        out = out + "#MSUB -V\n\n"
        out = out + "srun -n" + str(self.np) + " " + commandline
        return out
    def runscript(self,commandline):
        (handle,tempname) = tempfile.mkstemp()
        fd = os.fdopen(handle,"w")
        fd.write(self.getscript(commandline))
        fd.close()
        thisdir = os.getcwd()
        stdout = ""
        try:
            os.chdir(self.workingdir)
            # print("Running in",self.workingdir)
            stdout = commands.getoutput("msub '" + tempname + "'")
            os.chdir(thisdir)
            outputfilename = "slurm-" + str(int(stdout)) + ".out"
            # print("  " + outputfilename)
        except OSError:
            print("Bad directory:",self.workingdir)
        os.remove(tempname)
        return self.workingdir + outputfilename

def main():
    print("Little module to help automate running jobs on LC machines.")

if __name__ == "__main__":
    main()
