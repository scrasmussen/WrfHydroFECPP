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
import time as timer
import os
from os import path

#this script requires hgrid.utm, hgrid.ll, sflux2source.prop, and netcdf precipitation file
#this path should include all of those files; if it doesn't, paths will need to be adjusted below
result = []


def is_needed(preprocess_path):
    delaunay_file    = path.join(preprocess_path, 'delaunay.pre')
    area_cor_file    = path.join(preprocess_path, 'area_cor.pre')
    simplex_file     = path.join(preprocess_path, 'simplex.pre')
    precip2flux_file = path.join(preprocess_path, 'precip2flux.pre')
    file_list = [delaunay_file, area_cor_file, simplex_file, precip2flux_file]
    result = list(map(path.exists, file_list))
    if all(result):
        return False
    else:
        return True


# # --------------------
# # ---- preprocess ----
# # --------------------
def preprocess_files(input_path, preprocess_path, points, lat, lon):
    delaunay_file    = path.join(preprocess_path, 'delaunay.pre')
    precip2flux_file = path.join(preprocess_path, 'precip2flux.pre')
    simplex_file     = path.join(preprocess_path, 'simplex.pre')
    prop_file        = path.join(preprocess_path, 'propPoints.npy')
    elem_file        = path.join(preprocess_path, 'elem.pre')
    area_cor_file    = path.join(preprocess_path, 'area_cor.pre')


    t0 = timer.perf_counter()
    t = process_delaunay(delaunay_file, points)
    t1   = timer.perf_counter(); print(f"took {t1 - t0:0.4f} seconds")

    precip2flux, elem = process_precip2flux_and_elem(precip2flux_file, elem_file, input_path)
    t2   = timer.perf_counter(); print(f"took {t2 - t1:0.4f} seconds")

    simplex, propPoints = process_simplex(simplex_file, prop_file, input_path, t, elem)
    t3   = timer.perf_counter(); print(f"took {t3 - t2:0.4f} seconds")

    area_cor = process_area_cor(area_cor_file, lat, lon, t, simplex, propPoints)
    t4   = timer.perf_counter(); print(f"took {t4 - t3:0.4f} seconds")

    return t, precip2flux, simplex, area_cor, len(elem)

    # start = timer.perf_counter()
    # print("Closing Delaunay triangulation file")
    # fin   = timer.perf_counter()
    # print(f"took {fin - start:0.4f} seconds")

def process_precip2flux_and_elem(precip2flux_file, elem_file, input_path):
    if path.exists(precip2flux_file) and path.exists(elem_file):
        print("Reading precip2flux and elem from file")
        precip2flux_f = open(precip2flux_file, 'rb')
        precip2flux = pickle.load(precip2flux_f)
        precip2flux_f.close()

        elem_f = open(elem_file, 'rb')
        elem = pickle.load(elem_f)
        elem_f.close()
        return precip2flux, elem
    print("Calculating precip2flux and elem")
    #read in hgrid.utm to get x,y,z in meters and list of elements
    x = []
    y = []
    z = []
    elem = []
    with open(path.join(input_path, 'hgrid.utm')) as f:
        f.readline()
        line = f.readline()
        ne = int(line.split()[0])
        nn = int(line.split()[1])
        for i in range(nn):
            line = f.readline()
            x.append(float(line.split()[1]))
            y.append(float(line.split()[2]))
            z.append(float(line.split()[3]))
        for i in range(ne):
            line = f.readline()
            elem.append([int(line.split()[2]),int(line.split()[3]),int(line.split()[4])])
    #calculate the area of each element and multiply by 1/density of water for use later
    precip2flux = np.zeros((ne,1))
    for i in range(ne):
        e = elem[i]
        precip2flux[i] = (np.abs((x[e[2]-1]*y[e[1]-1]+x[e[1]-1]*y[e[0]-1]+x[e[0]-1]*y[e[2]-1])-(y[e[2]-1]*x[e[1]-1]+y[e[1]-1]*x[e[0]-1]+y[e[0]-1]*x[e[2]-1]))/2)/1000

    precip2flux_f = open(precip2flux_file, 'wb')
    pickle.dump(precip2flux, precip2flux_f)
    precip2flux_f.close()
    elem_f = open(elem_file, 'wb')
    pickle.dump(elem, elem_f)
    elem_f.close()
    return precip2flux, elem

def process_simplex(simplex_file, prop_file, input_path, t, elem):
    if path.exists(simplex_file) and path.exists(prop_file):
        print("Reading simplex from file")
        simplex_f = open(simplex_file, 'rb')
        simplex = pickle.load(simplex_f)
        simplex_f.close()
        # read in propPoints to get avgX, avgY
        with open(prop_file, 'rb') as f:
            propPoints = np.load(f)
        return simplex, propPoints
    print("Calculating Simplex and propPoints")

    #read in hgrid.ll to get x,y in degrees
    x = []
    y = []
    print("Reading hgrid.ll from file and calc")
    start = timer.perf_counter()

    dims_f = open(path.join(input_path, 'hgrid.utm'))
    dims_f.readline()
    line = dims_f.readline()
    ne = int(line.split()[0])
    nn = int(line.split()[1])
    dims_f.close()

    with open(path.join(input_path, 'hgrid.ll')) as f:
        f.readline()
        f.readline()
        for i in range(nn):
            line = f.readline()
            x.append(float(line.split()[1]))
            y.append(float(line.split()[2]))
    #calculate the center of each triangle by finding the average lat,lon for each element
    avgX = []
    avgY = []
    for i in range(ne):
        e = elem[i]
        avgX.append((x[e[0]-1]+x[e[1]-1]+x[e[2]-1])/3)
        avgY.append((y[e[0]-1]+y[e[1]-1]+y[e[2]-1])/3)

        #create list of center points for relevant elements
    propPoints = []
    for i in range(len(avgX)):
        propPoints.append([avgX[i],avgY[i]])
    propPoints = np.array(propPoints)

    #returning to Dalaunay triangulation workflow
    simplex = t.find_simplex(propPoints)

    print("Writing simplex to file")
    simplex_f = open(simplex_file, 'wb')
    pickle.dump(simplex, simplex_f)
    simplex_f.close()

    print("Writing propPoints to file")
    with open(prop_file, 'wb') as f:
        np.save(f, propPoints)
    return simplex, propPoints


def process_area_cor(area_cor_file, lat, lon, t, simplex, propPoints):
    if path.exists(area_cor_file):
        print("Reading Area Cor points from file")
        area_cor_f = open(area_cor_file, 'rb')
        area_cor = pickle.load(area_cor_f)
        area_cor_f.close()
        return area_cor
    print("Calculating Area Cor")

    avgX, avgY = np.split(propPoints,2,axis=1)
    avgX = avgX.flatten()
    avgY = avgY.flatten()

    area_lat = lat[t.simplices[simplex,:]]
    area_lon = lon[t.simplices[simplex,:]]
    # print("types::", type(area_lat),type(area_lon))



    arealatlon = [np.abs((area_lon[i,2]*area_lat[i,1]+area_lon[i,1]*area_lat[i,0]+area_lon[i,0]*area_lat[i,2])-(area_lat[i,2]*area_lon[i,1]+area_lat[i,1]*area_lon[i,0]+area_lat[i,0]*area_lon[i,2]))/2 for i in range(len(area_lat))]
    area_cor = np.zeros((area_lat.shape[0],area_lat.shape[1]))
    seq3 = [0,1,2,0,1]
    for k in range(3):
        area_lat0 = [avgY,list(lat[t.simplices[simplex,seq3[k+1]]].data),list(lat[t.simplices[simplex,seq3[k+2]]].data)]
        area_lon0 = [avgX,list(lon[t.simplices[simplex,seq3[k+1]]].data),list(lon[t.simplices[simplex,seq3[k+2]]].data)]
        arealatlon0 = [np.abs((area_lon0[2][i]*area_lat0[1][i]+area_lon0[1][i]*area_lat0[0][i]+area_lon0[0][i]*area_lat0[2][i])-(area_lat0[2][i]*area_lon0[1][i]+area_lat0[1][i]*area_lon0[0][i]+area_lat0[0][i]*area_lon0[2][i]))/2 for i in range(len(area_lat))]
        # arealatlon0 = []
        # for i in range(len(area_lat)):
        #     print("-------")
        #     # print(area_lon0[1][i]*area_lat0[0][i],'=',area_lon0[1][i],'*',area_lat0[0][i])
        #     print(np.shape(area_lat0), np.shape(area_lon0))
        #     print("======")
        #     print(area_lon0[0])
        #     # print("|||||||||||")
        #     # print(area_lon0[1][0:10])
        #     # print("|||||||||||")
        #     # print(area_lon0[2][0:10])
        #     # print("|||||||||||")
        #     # arealatlon0[i] = 1
        #     print("======")
        #     arealatlon0.append(np.abs((area_lon0[2][i]*area_lat0[1][i] +
        #                              area_lon0[1][i]*area_lat0[0][i] + # TypeError:
        #                              # 'builtin_function_or_method' object is not subscriptable
        #                              area_lon0[0][i]*area_lat0[2][i])-
        #                             (area_lat0[2][i]*area_lon0[1][i] +
        #                              area_lat0[1][i]*area_lon0[0][i] +
        #                              area_lat0[0][i]*area_lon0[2][i])) / 2)

        #     sys.exit()

        # Runtime: invalid value encourtered in reduce, divid by zero encountered in double_scalars
        area_cor[:,k] = list(map(truediv,arealatlon0,arealatlon))

    area_cor_f = open(area_cor_file, 'wb')
    pickle.dump(area_cor, area_cor_f)
    area_cor_f.close()
    return area_cor

#read in Delaunay triangulation data if it exists or calculate and save data if it does not
def process_delaunay(delaunay_file, points):
    if path.exists(delaunay_file):
        print("Reading Delaunay triangulation points from file")
        delaunay_f = open(delaunay_file, 'rb')
        t = pickle.load(delaunay_f)
    else:
        #perform delaunay triangulation on preciptiation points
        #save output to file for future runs
        print("Calculating Delaunay triangulation points and writing to file")
        t = Delaunay(points)
        delaunay_f = open(delaunay_file, 'wb')
        pickle.dump(t, delaunay_f)
    delaunay_f.close()
    return t
