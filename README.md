# GameOfLife
Implementation of cellular automaton called the (Conway's) Game of Life using Python3 and Open MPI 4.0.2.

Requirements to run this program:
Python3 and Open MPI 4.0.2

How to run on Linux System:
1. Open terminal in the directory of project files.
2. Run "mpirun -np M --oversubscribe python3 game.py input.txt output.txt T" command, where M is number of processors to run with and T is number of iterations for the game.

Note that M should (i^2) + 1, where i is an even number.

input.txt is input file which is a (N x N) 2-dimensional array.

output.txt is output file which is a (N x N) 2-dimensional array.
