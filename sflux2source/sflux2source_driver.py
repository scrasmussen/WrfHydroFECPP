#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 18:01:45 2021

@author: Camaron.George
@editor: Soren Rasmussen (NCAR)
"""
# Changes: Delaunay triangulation either read from file or calculated and saved. -Soren
# Changes: vsource.th.2 output uses MPI to compute and communicated back to main to write -Soren
import numpy as np
import netCDF4 as nc
from mpi4py import MPI
from operator import truediv
from scipy.spatial import Delaunay
import pickle
import time as timer
import sflux2source_preprocess as preprocess
from multiprocessing import Process
from sflux2source_output import write_vsource, write_source_sink, write_msource, write_vsource_mpi
from multiprocessing import Process
from os import environ, path, mkdir
import sys


def main_comp(input_path, outfile1, outfile2, outfile3, run_MPI):
    # timer
    print("Reading in precip.nc")
    start = timer.perf_counter()

    #read in precipitation data
    data = nc.Dataset(path.join(input_path, file))
    lon = data.variables['lon'][:]
    lon = lon.flatten()
    lat = data.variables['lat'][:]
    lat = lat.flatten()
    time = data.variables['time'][:]
    time = time*86400
    dt = time[1]-time[0]
    prate = data.variables['prate'][:]

    #create list of points in precipitation file
    points = list(zip(lon, lat))

    # end timer
    fin   = timer.perf_counter()
    print(f"took {fin - start:0.4f} seconds")


    # ---- preprocess ----
    if preprocess.is_needed(preprocess_path):
        print("Preprocessing is needed")
    else:
        print("Reading in Preprocessed Files")
    t, area_cor, simplex, precip2flux, len_elem = preprocess.preprocess_files(input_path, preprocess_path,
                                                                    points, lat, lon)

    # ----- workflow -----
    # Static: t (Delaunay points), area_cor, simplex, precip2flux
    # Dynamic: prate, time
    if (run_MPI):
        time = data.variables['time'][:]
        return prate, area_cor, precip2flux, time, t.simplices, simplex, len_elem
    else:
        p1 = Process(target=write_vsource, args=(outfile1, prate, area_cor, precip2flux, time, t, simplex,))

        p2 = Process(target=write_source_sink, args=(outfile2, len_elem,))

        p3 = Process(target=write_msource, args=(outfile3, time, len(time), len_elem,))

        start = timer.perf_counter()
        p1.start()
        p2.start()
        p3.start()

        p2.join()
        p3.join()
        p1.join()
        fin = timer.perf_counter()
        print("Writing output files took",str(fin-start))
    # old serial version
    # else:
    #     start = timer.perf_counter()
    #     o_start = timer.perf_counter()
    #     write_vsource(path.join(output_path, outfile1), prate, area_cor, precip2flux, time, t, simplex)
    #     fin   = timer.perf_counter()
    #     print(f"took {fin - start:0.4f} seconds")

    #     print("Writing output to", outfile2)
    #     start = timer.perf_counter()
    #     write_source_sink(path.join(output_path, outfile2), len_elem)
    #     fin   = timer.perf_counter()
    #     print(f"took {fin - start:0.4f} seconds")

    #     print("Writing output to", outfile3)
    #     start = timer.perf_counter()
    #     write_msource(path.join(output_path, outfile3), time, len(time), len_elem)
    #     fin = timer.perf_counter()
    #     print(f"took {fin - start:0.4f} seconds")
    #     print(f"total output time took {fin - o_start:0.4f} seconds")



if __name__ == '__main__':
    beginning = timer.perf_counter()
    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank

    #this script requires hgrid.utm, hgrid.ll, sflux2source.prop, and netcdf precipitation file
    #this path should include all of those files; if not, paths will need to be adjusted below
    input_path = environ['COASTAL_WORK_DIR']
    output_path = input_path
    output_path = 'output_mpi'

    if not path.exists(output_path) and (rank == 0):
        mkdir(output_path)


    preprocess_path = input_path + '/preprocessed'
    file = 'precip.nc'
    outfile1 = path.join(output_path, 'vsource.th.2')
    outfile2 = path.join(output_path, 'source_sink.in.2')
    outfile3 = path.join(output_path, 'msource.th.2')

    if (size == 1):
        print("SIZE IS 1")
        sys.exit()
        main_comp(run_MPI=False)
        sys.exit()
    elif (size > 1 and rank == 0):
        print("Running MPI!")
        prate, area_cor, precip2flux, time, simplices, simplex, len_elem = \
                                                                      main_comp(input_path, outfile1,
                                                                                outfile2, outfile3,
                                                                                run_MPI=True)

    elif (size > 1 and rank != 0):
        len_elem = None
        prate = None
        area_cor = None
        precip2flux = None
        time = None
        simplices = None
        simplex = None

    start = timer.perf_counter()
    len_elem = comm.bcast(len_elem, root=0)
    prate = comm.bcast(prate, root=0)
    area_cor = comm.bcast(area_cor, root=0)
    precip2flux = comm.bcast(precip2flux, root=0)
    time = comm.bcast(time, root=0)
    time = time*86400 # had to send smaller time, fixing
    simplices = comm.bcast(simplices, root=0)
    simplex = comm.bcast(simplex, root=0)
    if (rank == 1):
        fin   = timer.perf_counter()
        print(f"broadcast took {fin - start:0.4f} seconds")

    start = timer.perf_counter()
    write_vsource_mpi(outfile1, prate, area_cor, precip2flux, time, simplices, simplex)
    if (rank == 1):
        fin   = timer.perf_counter()
        print(outfile1+f" took {fin - start:0.4f} seconds")


    # have non-root rank write outfile 2 and 3 if available
    if (rank == 1) or (size < 2):
        print("Writing output to outfile2")
        start = timer.perf_counter()
        write_source_sink(outfile2, len_elem)
        fin   = timer.perf_counter()
        print(outfile2+f" took {fin - start:0.4f} seconds")

    if (rank == 2) or (size < 2):
        print("Writing output to outfile3")
        start = timer.perf_counter()
        write_msource(outfile3, time, len(time), len_elem)
        fin = timer.perf_counter()
        print(outfile3+f" took {fin - start:0.4f} seconds")

    if rank == 0:
        fin = timer.perf_counter()
        print("-----------------")
        print("")
        print(f"Total time {fin - beginning:0.4f} seconds")
