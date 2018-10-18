#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:24:15 2018

@author: cham
"""


import bisect
import numpy as np


def grid_to_meshflat(*grids):
    """ convert a grid to meshflat arrays """
    # get dimensions
    Ns = [len(_) for _ in grids]
    # make indices of grids
    grids_ind = [np.arange(_, dtype=np.int) for _ in Ns]
    # make mesh_grids
    meshs = np.meshgrid(*grids)
    # make mesh_ind
    meshs_ind = np.meshgrid(*grids_ind)
    # make flat_mesh_grids
    flats = np.array([_.flatten() for _ in meshs]).T
    # make flat_mesh_ind
    flats_ind = np.array([_.flatten() for _ in meshs_ind]).T

    # make a dict of flat_mesh_ind:i
    ind_dict = dict()
    for i in range(len(flats_ind)):
        ind_dict[tuple(flats_ind[i])] = i

    return grids, grids_ind, meshs, meshs_ind, flats, flats_ind, ind_dict


def bisect_interval(edges=[1,2,3], x=.1):
    """ return the nearest edges using bisect """
    if edges[0] <= x <= edges[-1]:
        _ = bisect.bisect_left(edges, x)
        return _-1, _
    else:
        # null value, returned when x is not in bounds
        return -9, -9


class RegularGrid():
    """  """
    def __init__(self, *grids, values=None):
        _ = grid_to_meshflat(*grids)
        self.grids = _[0]
        self.grids_ind = _[1]
        self.meshs = [2]
        self.meshs_ind = _[3]
        self.flats = _[4]
        self.flats_ind = _[5]
        self.ind_dict = _[6]
        self.values = values

    def set_values(self, values):
        self.values = np.array(values)

    def interp3(self, pos):

        e1 = bisect_interval(self.grids[0], pos[0])
        e2 = bisect_interval(self.grids[1], pos[1])
        e3 = bisect_interval(self.grids[2], pos[2])

        if e1[0] <-1 or e2[0] <-1 or e3[0] <-1:
            """ out of bounds """
            return None
        else:
            # calculate nodes
            p1_0 = self.grids[0][e1[0]]
            p1_1 = self.grids[0][e1[1]]
            p2_0 = self.grids[1][e2[0]]
            p2_1 = self.grids[1][e2[1]]
            p3_0 = self.grids[2][e3[0]]
            p3_1 = self.grids[2][e3[1]]

            v_tot = (p1_1-p1_0)*(p2_1-p2_0)*(p3_1-p3_0)
            v_000 = (p1_1-pos[0])*(p2_1-pos[1])*(p3_1-pos[2])
            v_100 = (pos[0]-p1_0)*(p2_1-pos[1])*(p3_1-pos[2])
            v_010 = (p1_1-pos[0])*(pos[1]-p2_0)*(p3_1-pos[2])
            v_110 = (pos[0]-p1_0)*(pos[1]-p2_0)*(p3_1-pos[2])
            v_001 = (p1_1-pos[0])*(p2_1-pos[1])*(pos[2]-p3_0)
            v_101 = (pos[0]-p1_0)*(p2_1-pos[1])*(pos[2]-p3_0)
            v_011 = (p1_1-pos[0])*(pos[1]-p2_0)*(pos[2]-p3_0)
            v_111 = (pos[0]-p1_0)*(pos[1]-p2_0)*(pos[2]-p3_0)
            #v_000+v_001+v_010+v_011+v_100+v_101+v_110+v_111

            i_000 = self.ind_dict[e1[0],e2[0],e3[0]]
            i_001 = self.ind_dict[e1[0],e2[0],e3[1]]
            i_010 = self.ind_dict[e1[0],e2[1],e3[0]]
            i_011 = self.ind_dict[e1[0],e2[1],e3[1]]
            i_100 = self.ind_dict[e1[1],e2[0],e3[0]]
            i_101 = self.ind_dict[e1[1],e2[0],e3[1]]
            i_110 = self.ind_dict[e1[1],e2[1],e3[0]]
            i_111 = self.ind_dict[e1[1],e2[1],e3[1]]

            w = np.array([v_000, v_001, v_010, v_011, v_100, v_101, v_110, v_111]).reshape(-1, 1)/v_tot
            #specs = spec[np.array([i_000, i_001, i_010, i_011, i_100, i_101, i_110, i_111])]
            #figure();plot(spec_wm);plot(specs.T)
            spec_wm = np.sum(self.values[np.array([i_000, i_001, i_010, i_011, i_100, i_101, i_110, i_111])]*w, axis=0)

            return spec_wm
