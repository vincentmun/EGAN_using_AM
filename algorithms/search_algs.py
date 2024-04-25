"""GanAlgorithm."""
from __future__ import absolute_import, division, print_function
import copy
import os
import numpy as np
import random


# To implement easy, wo use int number in the code to represent the
# corresponding operations. These are replaced with one-hot in the paper,
# which can better explain our theory. Each int number is equal to a one-hot number.


class GanAlgorithm():

    def __init__(self, args):
        self.steps = 3
        self.up_nodes = 2
        self.down_nodes = 2
        self.normal_nodes = 5
        self.dis_normal_nodes = 5
        self.archs = {}
        self.dis_archs = {}
        self.base_dis_arch = np.array(
            [[3, 0, 3, 0, 2, 0, 0], [3, 0, 3, 0, 1, -1, -1], [3, 0, 3, 0, 1, -1, -1]])
        self.Normal_G = []
        self.Up_G = []
        self.Normal_G_fixed = []
        self.Up_G_fixed = []
        self.Normal_G = [[(op, 0) for op in range(7)] for _ in range(self.steps * self.normal_nodes)]
        self.Up_G = [[(op, 0) for op in range(3)] for _ in range(self.steps * self.up_nodes)]
            
        initial_weight = 1.0  # Default initial weight
        for i in range(self.steps * self.normal_nodes):
            self.Normal_G_fixed.append([(op, initial_weight) for op in range(7)])  # 7 normal operations

        for i in range(self.steps * self.up_nodes):
            self.Up_G_fixed.append([(op, initial_weight) for op in range(3)])  # 3 up operations
        self.use_weights = False
        
        
    def set_use_weights(self, value):
        self.use_weights = value

    def Get_Operation(self, Candidate_Operation, mode, num, remove=True):
        '''
        Candidate_Operation: operation pool
        mode: Operation type
        return: the operation
        remove: remove from the pool
        '''
        if Candidate_Operation == [] and mode == 'up':
            Candidate_Operation += self.Up_G_fixed[num]
        elif Candidate_Operation == [] and mode == 'normal':
            Candidate_Operation += self.Normal_G_fixed[num]

        if self.use_weights:
            operations, weights = zip(*Candidate_Operation)
            choice = random.choices(operations, weights=weights)[0]
        else:
            choice = random.choice([op[0] for op in Candidate_Operation])

        if remove:
            Candidate_Operation = [op_tuple for op_tuple in Candidate_Operation if op_tuple[0] != choice]

        return choice
    

    def sample_fair(self, remove=True):
        genotype = np.zeros(
            (self.steps, self.up_nodes + self.normal_nodes), dtype=int)  # 3*7
        for i in range(self.steps):
            for j in range(self.up_nodes):
                genotype[i][j] = self.Get_Operation(self.Up_G[2 * i + j], 'up', 2 * i + j, remove)

            for k in range(2):
                genotype[i][k + 2] = self.Get_Operation(self.Normal_G[5 * i + k], 'normal', 5 * i + k, remove)
            while (genotype[i][2] == 0 and genotype[i][3] == 0):
                for k in range(2):
                    genotype[i][k + 2] = self.Get_Operation(self.Normal_G[5 * i + k], 'normal', 5 * i + k, False)

            for k in range(2, self.normal_nodes):
                genotype[i][k + 2] = self.Get_Operation(self.Normal_G[5 * i + k], 'normal', 5 * i + k, remove)
            while (genotype[i][4] == 0 and genotype[i][5] == 0 and genotype[i][6] == 0):
                for k in range(2, self.normal_nodes):
                    genotype[i][k + 2] = self.Get_Operation(self.Normal_G[5 * i + k], 'normal', 5 * i + k, False)

        return genotype

    def encode(self, genotype):
        lists = [0 for i in range(self.steps)]
        for i in range(len(lists)):
            lists[i] = str(genotype[i])
        return tuple(lists)

    def search(self, remove=True):
        new_genotype = self.sample_fair(remove)
        return new_genotype