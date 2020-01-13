
#Status : Compiling and Working
#Implementation with Checkered Splits and Periodic

import sys
import numpy as np
import math
from mpi4py import MPI

comm = MPI.COMM_WORLD  #create MPI environment
rank = comm.Get_rank()  #rank of a process
size = comm.Get_size()  #number of total processes

boardDim = 360  #dimension of input board
worker_count = size - 1     #number of workers
sliceCount = worker_count   #number of slices
sliceDimension = boardDim / math.sqrt(sliceCount)   #dimension of each slice
sliceDimension = int(sliceDimension)
sqrtW = int(math.sqrt(worker_count))       #how many ranks on each row and column

firstBoard = np.empty((boardDim, boardDim), dtype=int)  #initialize first board

if rank == 0:

    inputFile = sys.argv[1]     #name of input file
    firstBoard = np.loadtxt(inputFile, dtype=int)  #store initial board
   #checkBoard = np.loadtxt('gliderCheck.txt', dtype=int)   #for checking test cases

    sliceArr = np.empty((sliceDimension, sliceDimension), dtype=int)
    sliceNum = 1
    for i in range(1, sqrtW + 1):       #split the board and distribute them to workers
        for j in range(1, sqrtW + 1):
            for k in range(0, sliceDimension):
                for m in range(0, sliceDimension):
                    sliceArr[k][m] = firstBoard[k + ((i - 1) * sliceDimension)][m + ((j - 1) * sliceDimension)]
            comm.send(sliceArr, dest=sliceNum)
            sliceNum += 1
else:
    recvSlice = comm.recv(source=0)     # receive slice

iteration = int(sys.argv[3])     #iteration number
for it in range(iteration):      #iteration for loop

    if rank != 0:  #worker processes communicate
        #Share bottom row
        if (int((rank - 1) / sqrtW) % 2) == 0:
            if rank + sqrtW > worker_count:
                comm.send(recvSlice[sliceDimension - 1], dest=((rank - 1) % sqrtW) + 1)
            else:
                comm.send(recvSlice[sliceDimension - 1], dest=rank + sqrtW)
        else:
            if rank <= sqrtW:
                topArr = comm.recv(source=sqrtW * (sqrtW - 1) + rank)
            else:
                topArr = comm.recv(source=rank - sqrtW)

        if (int((rank - 1) / sqrtW) % 2) == 1:
            if rank + sqrtW > worker_count:
                comm.send(recvSlice[sliceDimension - 1], dest=((rank - 1) % sqrtW) + 1)
            else:
                comm.send(recvSlice[sliceDimension - 1], dest=rank + sqrtW)
        else:
            if rank <= sqrtW:
                topArr = comm.recv(source=sqrtW * (sqrtW - 1) + rank)
            else:
                topArr = comm.recv(source=rank - sqrtW)
        # End of sharing bottom row

        #Share top row
        if (int((rank - 1) / sqrtW) % 2) == 0:
            if rank <= sqrtW:
                comm.send(recvSlice[0, :], dest=(sqrtW * (sqrtW - 1)) + rank)
            else:
                comm.send(recvSlice[0, :], dest=(rank - sqrtW))
        else:
            if rank + sqrtW > worker_count:
                botArr = comm.recv(source=((rank - 1) % sqrtW) + 1)
            else:
                botArr = comm.recv(source=rank + sqrtW)

        if (int((rank - 1) / sqrtW) % 2) == 1:
            comm.send(recvSlice[0, :], dest=(rank - sqrtW))
        else:
            botArr = comm.recv(source=rank + sqrtW)
        #End of sharing top row

        #Share right most column
        if (rank % 2) == 1:
            comm.send(recvSlice[:, sliceDimension - 1], dest=(rank + 1))
        else:
            leftArr = comm.recv(source=(rank - 1))

        if (rank % 2) == 0:
            if (rank % sqrtW) == 0:
                comm.send(recvSlice[:, sliceDimension - 1], dest=(rank + 1 - sqrtW))
            else:
                comm.send(recvSlice[:, sliceDimension - 1], dest=rank + 1)
        else:
            if (rank % sqrtW) == 1:
                leftArr = comm.recv(source=(rank + sqrtW - 1))
            else:
                leftArr = comm.recv(source=(rank - 1))
        #End of sharing right most column

        #Share left most column
        if (rank % 2) == 1:
            if (rank % sqrtW) == 1:
                comm.send(recvSlice[:, 0], dest=(rank + sqrtW - 1))
            else:
                comm.send(recvSlice[:, 0], dest=(rank - 1))

        else:
            if (rank % sqrtW) == 0:
                rightArr = comm.recv(source=(rank - sqrtW + 1))
            else:
                rightArr = comm.recv(source=(rank + 1))

        if (rank % 2) == 0:
            comm.send(recvSlice[:, 0], dest=(rank - 1))
        else:
            rightArr = comm.recv(source=(rank + 1))
        #End of sharing left most column

        #Share bottom right
        if (rank % 2) == 1:  #odd ranks share
            if rank + sqrtW > worker_count:
                comm.send(recvSlice[sliceDimension - 1, sliceDimension - 1], dest=(rank % sqrtW) + 1)
            else:
                comm.send(recvSlice[sliceDimension - 1, sliceDimension - 1], dest=(rank + 1 + sqrtW))
        else:
            if rank <= sqrtW:
                topLeft = comm.recv(source=rank + (sqrtW * (sqrtW - 1)) - 1)

            else:
                topLeft = comm.recv(source=rank - (sqrtW + 1))

        if (rank % 2) == 0: #even ranks share
            if (rank % sqrtW) == 0:
                if rank == worker_count:
                    comm.send(recvSlice[sliceDimension - 1, sliceDimension - 1], dest=1)
                else:
                    comm.send(recvSlice[sliceDimension - 1, sliceDimension - 1], dest=(rank + 1))
            else:
                if rank + sqrtW > worker_count:
                    comm.send(recvSlice[sliceDimension - 1, sliceDimension - 1], dest=rank - (sqrtW * (sqrtW - 1)) + 1)
                else:
                    comm.send(recvSlice[sliceDimension - 1, sliceDimension - 1], dest=rank + sqrtW + 1)
        else:
            if (rank % sqrtW) == 1:
                if rank == 1:
                    topLeft = comm.recv(source=worker_count)

                else:
                    topLeft = comm.recv(source=rank - 1)

            else:
                if rank <= sqrtW:
                    topLeft = comm.recv(source=rank + (sqrtW * (sqrtW - 1)) - 1)
                else:
                    topLeft = comm.recv(source=rank - sqrtW - 1)
        #End of sharing bottom right

        #Share top left
        if (rank % 2) == 1:  #Odd ranks share
            if rank <= sqrtW:
                if rank == 1:
                    comm.send(recvSlice[0, 0], dest=worker_count)
                else:
                    comm.send(recvSlice[0, 0], dest=rank + (sqrtW * (sqrtW - 1)) - 1)
            else:
                if rank % sqrtW == 1:
                    comm.send(recvSlice[0, 0], dest=rank - 1)
                else:
                    comm.send(recvSlice[0, 0], dest=rank - sqrtW - 1)
        else:
            if rank % sqrtW == 0:
                if rank == worker_count:
                    botRight = comm.recv(source=1)

                else:
                    botRight = comm.recv(source=rank + 1)

            else:
                if rank + sqrtW > worker_count:
                    botRight = comm.recv(source=rank - (sqrtW * (sqrtW - 1)) + 1)

                else:
                    botRight = comm.recv(source=rank + sqrtW + 1)

        if (rank % 2) == 0: #even ranks share
            if rank <= sqrtW:

                comm.send(recvSlice[0, 0], dest=rank + (sqrtW * (sqrtW - 1)) - 1)

            else:
                comm.send(recvSlice[0, 0], dest=rank - sqrtW - 1)

        else:
            if rank + sqrtW > worker_count:
                botRight = comm.recv(source=rank - (sqrtW * (sqrtW - 1)) + 1)

            else:
                botRight = comm.recv(source=rank + sqrtW + 1)
        #End of sharing top left

        #Sharing top right
        if (rank % 2) == 1:  #odd ranks share
            if rank <= sqrtW:
                comm.send(recvSlice[0][sliceDimension - 1], dest=sqrtW * (sqrtW - 1) + rank + 1)
            else:
                comm.send(recvSlice[0][sliceDimension - 1], dest=rank - sqrtW + 1)
        else:
            if rank + sqrtW > worker_count:
                botLeft = comm.recv(source=rank - (sqrtW * (sqrtW - 1)) - 1)

            else:
                botLeft = comm.recv(source=rank + sqrtW - 1)

        if (rank % 2) == 0: #even ranks share
            if rank <= sqrtW:
                if rank == sqrtW:
                    comm.send(recvSlice[0, sliceDimension - 1], dest=sqrtW * (sqrtW - 1) + 1)
                else:
                    comm.send(recvSlice[0, sliceDimension - 1], dest=sqrtW * (sqrtW - 1) + rank + 1)
            elif rank % sqrtW == 0:
                comm.send(recvSlice[0, sliceDimension - 1], dest=rank - 2 * sqrtW + 1)
            else:
                comm.send(recvSlice[0, sliceDimension - 1], dest=rank - sqrtW + 1)

        else:
            if rank + sqrtW > worker_count:
                if rank == (sqrtW * (sqrtW - 1) + 1):
                    botLeft = comm.recv(source=sqrtW)

                else:
                    botLeft = comm.recv(source=rank - (sqrtW * (sqrtW - 1)) - 1)

            else:
                if rank % sqrtW == 1:
                    botLeft = comm.recv(source=rank + (2 * sqrtW) - 1)

                else:
                    botLeft = comm.recv(source=rank + sqrtW - 1)
        #End of sharing top right

        #Share bottom left
        if (rank % 2) == 1: #odd numbers send
            if rank + sqrtW > worker_count:
                if rank == (sqrtW * (sqrtW - 1)) + 1:
                    comm.send(recvSlice[sliceDimension - 1][0], dest=sqrtW)
                else:
                    comm.send(recvSlice[sliceDimension - 1][0], dest=rank - (sqrtW * (sqrtW - 1)) - 1)
            elif rank % sqrtW == 1:
                comm.send(recvSlice[sliceDimension - 1][0], dest=rank + (2 * sqrtW) - 1)
            else:
                comm.send(recvSlice[sliceDimension - 1][0], dest=rank + sqrtW - 1)
        else:
            if rank <= sqrtW:
                if rank == sqrtW:
                    topRight = comm.recv(source=sqrtW * (sqrtW - 1) + 1)

                else:
                    topRight = comm.recv(source=rank + (sqrtW * (sqrtW - 1)) + 1)

            else:
                if rank % sqrtW == 0:
                    topRight = comm.recv(source=rank - (2 * sqrtW) + 1)

                else:
                    topRight = comm.recv(source=rank - sqrtW + 1)

        if (rank % 2) == 0: #even numbers send
            if rank + sqrtW > worker_count:
                comm.send(recvSlice[sliceDimension - 1, 0], dest=rank - (sqrtW * (sqrtW - 1)) - 1)
            else:
                comm.send(recvSlice[sliceDimension - 1, 0], dest=rank + sqrtW - 1)
        else:
            if rank < sqrtW:
                topRight = comm.recv(source=rank + (sqrtW * (sqrtW - 1)) + 1)

            else:
                topRight = comm.recv(source=rank - sqrtW + 1)
        #End of sharing bottom left

    if rank != 0:
        tempArr = np.empty((sliceDimension, sliceDimension), dtype=int) # temporary array for holding next state of cells
        tempArr = recvSlice

        bigDim = len(recvSlice) + 2
        bigArr = np.empty((bigDim, bigDim), dtype=int)  #initialize bigger array with larger dimension
        bigArr.fill(0)

        bigArr[1:recvSlice.shape[0] + 1, 1:recvSlice.shape[1] + 1] = recvSlice  #store the main array and necessary neighbors to the big array
        bigArr[1:len(leftArr) + 1, 0] = leftArr
        bigArr[0, 1:len(topArr) + 1] = topArr
        bigArr[1:len(rightArr) + 1, bigArr.shape[1] - 1] = rightArr
        bigArr[bigArr.shape[1] - 1, 1:len(botArr) + 1] = botArr

        bigArr[0][0] = topLeft
        bigArr[0][bigArr.shape[1] - 1] = topRight
        bigArr[bigArr.shape[1] - 1][bigArr.shape[1] - 1] = botRight
        bigArr[bigArr.shape[1] - 1][0] = botLeft

        count = 0
        for i in range(1, bigDim - 1):  #traverse over big array
            for j in range(1, bigDim - 1):
                count += 1
                neighSum = bigArr[i - 1][j - 1] + bigArr[i - 1][j] + bigArr[i - 1][j + 1] + bigArr[i][j - 1] + \
                           bigArr[i][j + 1] + bigArr[i + 1][j - 1] + bigArr[i + 1][j] + bigArr[i + 1][j + 1]

                if bigArr[i][j] == 0:
                    if neighSum == 3:   # if a 0 cell has exactly 3 alive neighbors then it will come to life
                        tempArr[i - 1][j - 1] = 1  #store the next state of a cell in tempArr

                elif bigArr[i][j] == 1:
                    if (neighSum < 2) | (neighSum > 3):  #if a cell is lonely or has too much neighbors it will die
                        tempArr[i - 1][j - 1] = 0  #store the next state of a cell in tempArr
        recvSlice = tempArr #make recvSlice to be tempArr

if rank != 0:
    comm.send(recvSlice, dest=0)  #send final state slice to master process
else:

    finalBoard = np.empty((boardDim, boardDim), dtype=int)  #create final board
    m = 1
    for i in range(1, sqrtW + 1):  # put received final slices from workers together
        for j in range(1, sqrtW + 1):
            finalBoard[(i - 1) * sliceDimension: sliceDimension*i, (j - 1) * sliceDimension: j * sliceDimension] = comm.recv(source=m)
            m += 1
    #print(np.array_equal(finalBoard, checkBoard))
    outputFile = sys.argv[2]  # name of output file
    np.savetxt(outputFile, finalBoard, fmt='%i')  #store final board to output file
