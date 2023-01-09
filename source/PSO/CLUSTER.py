import numpy as np
# define the class cluster for each cluster
class cluster :# A class store the position of each atom in a cluster and the energy of the whole cluster
    def dist_matric2(self):  
        #input: (n,3)matrix of n atoms
        #output: (n,n) matrix which m[i][j] means the distance the atom [i], and atom[j]  
        dis_mat=np.zeros((self.num_of_atom,self.num_of_atom))
        for def_i in range(self.num_of_atom):
            for def_j in range(self.num_of_atom):
                dis_mat[def_i][def_j] = np.sqrt(np.sum(np.square(self.pos[def_i]-self.pos[def_j])))
        return(dis_mat)
    
    def __init__ (self, line)->None:
        self.num_of_atom = int(line[0]) # number of atoms in the cluster
        self.pos = np.array(line[1:-1]).reshape((-1,3)) # a (n,3) matrix for each atom
        self.energy = line[-1] # the energy of the whole cluster
        self.dis_mat = self.dist_matric2() # the dis_matrix, which dis_mat[i][j] is the distance between atom[i] and atom[j] is pos
        assert(self.pos.shape[0]==self.num_of_atom)
        assert(self.pos.shape[1]==3)
        # print("this cluster has {} atoms\n there position is {}\n it is energy is{}\n, dis_mat is \n{}".format(self.num_of_atom, self.pos,self.energy,self.dis_mat)) # for test

