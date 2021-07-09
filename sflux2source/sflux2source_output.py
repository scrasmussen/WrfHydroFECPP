#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 18:01:45 2021

@author: Camaron.George
@editor: Soren Rasmussen (NCAR)
"""
# Changes: Delaunay triangulation either read from file or calculated and saved. -Soren
import numpy as np
import netCDF4 as nc
from operator import truediv
from scipy.spatial import Delaunay
import pickle
from mpi4py import MPI
from math import ceil, floor
import time as timer

import os
from os import path

from multiprocessing import Process


def write_vsource_mpi(f_name, prate, area_cor, precip2flux, time, simplices, simplex):
    # get range
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    n = len(time)



    split = np.array_split(range(18),size)
    start = split[rank][0]
    end = split[rank][-1]
    chunk = end - start + 1

    # compute
    A = []
    # sendbuf = np.zeros()

    for i in range(start,start+chunk):
        thisPrate = prate[i,:,:].flatten()
        thisPrate = thisPrate[simplices[simplex,:]]
        thisPrate = np.sum(thisPrate*area_cor,axis=1)
        len_prate = len(thisPrate)
        thisPrate = [thisPrate[i]*precip2flux[i][0] for i in range(len_prate)]
        # thisPrate.insert(0,time[i]) # o.write(str(time[i])+' ')
        # len_prate = len(thisPrate)
        A.append(thisPrate)

    A = np.array(A, dtype=np.float64)

    sendbufrows = chunk
    sendbufrows = comm.gather(sendbufrows, root=0)

    if rank == 0:
        res = np.zeros((n,len_prate), dtype='f')
        j = 0 # start
        res[j:j+sendbufrows[0],:] = A
        j = sendbufrows[0]
        for i in range(1,size):
            if (sendbufrows[i] <= 0):
                continue
            recv = np.zeros((sendbufrows[i],len_prate), dtype=np.float64)
            comm.Recv(recv,source=i,tag=i)
            res[j:j+sendbufrows[i],:] = recv
            j += sendbufrows[i]
        o = open(f_name,'w')
        for i in range(n):
            o.write(str(time[i])+' ')
            for j in range(len_prate):
                o.write(str(res[i,j])+' ')
        o.write('\n')
        o.close()

    else:
        comm.Send(A,dest=0, tag=rank)


#convert precipitation to streamflow and write discharge files with precipitation only
def write_vsource(f_name, prate, area_cor, precip2flux, time, t, simplex):
    o = open(f_name,'w')
    print("Calc and writing output to", o.name)

    for i in range(len(time)):
        # if (i < 10):
        #     t1 = timer.perf_counter()
        o.write(str(time[i])+' ')
        thisPrate = prate[i,:,:].flatten()
        thisPrate = thisPrate[t.simplices[simplex,:]]
        thisPrate = np.sum(thisPrate*area_cor,axis=1)
        thisPrate = [thisPrate[i]*precip2flux[i][0] for i in range(len(thisPrate))]
        # if (i < 10):
        #     t2 = timer.perf_counter()

        # takes ~40 seconds longer per iterations
        # np.savetxt(o, thisPrate, newline=" ")
        for j in range(len(thisPrate)):
            o.write(str(thisPrate[j])+' ')
        o.write('\n')
        # if (i < 10):
        #     t3 = timer.perf_counter()
        #     print("comp t = ", t2-t1, "write t =", t3-t2)


    o.close()
    print("Closing", o.name)


def write_source_sink(f_name, len_elem):
    o = open(f_name,'w')
    print("Writing output to", o.name)
    o.write(str(len_elem)+'\n')
    for i in range(1,len_elem+1):
        o.write(str(i)+'\n')
    o.write('\n'+'0')
    o.close()
    print("Closing", o.name)


def write_msource(f_name, time, len_time, len_elem):
    o = open(f_name,'w')
    print("Writing output to", o.name)
    for i in range(len_time):
        o.write(str(time[i]))
        for j in range(len_elem):
            o.write('\t-9999')
        o.write('\n0')
        for j in range(1,len_elem):
            o.write('\t0')
        o.write('\n')
    o.close()
    print("Closing", o.name)
