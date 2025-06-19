# GlueX-GenAnalysis
Generates code to make all possible basic plots for a given final state from a ROOT flat tree using RDataFrames.

This software is meant to be used in conjunction with my fork of the MakeDSelector program found [here](https://github.com/rmbrunner/gluex_root_analysis) as it generates the default list of branch names needed by this program. But, in principle, any flat ROOT tree should suffice.

Note that, by default, the program will attempt to auto-detect the name of the tree within the root file.

# Installation
For now I suggest in-directory installation (especially for JLab vdi users). Simply copy this projects files into your directory and run:
`make build_it`
Which will install the executable in a `bin/` folder in the current directory.

# Usage
`genAnalysis <input_ROOT_file> <branches.txt> <output_program.C>`

This will create a output_program.C ROOT macro that contains the implementation of all the plots. Note that "selectedBranches.txt" must contain only one branch name per line.

To run:
`root -l -b output_program.C`

# Warning
For complex final states the generated output will be *very* large. It may be relevant to trim down the default branches.txt file to a much smaller number.

The number of possible combinations of final state particles goes as n = 2ᴺ - 1 and the number of plots as A * ₙC₂ where A is the number of types of plots. 
