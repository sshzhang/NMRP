import numpy as np
import random
import time
import argparse


def parse_args():  # 包含两种元路径信息  ubub  ubcb
    parser = argparse.ArgumentParser(description="Run NMPRModel.")
    parser.add_argument('--walk_num', type=int, default=5,
                        help='the length of random walk .')
    parser.add_argument('--metapath', type=str, default="ubcb",
                        help='the metapath for amazon dataset.')
    return parser.parse_args()

class MetapathBasePathSample:
    def __init__(self, **kargs):
        self.metapath = kargs.get('metapath')
        # 每种元路径类型中抽取的具体实例路径数
        self.walk_num = kargs.get('walk_num')
        self.K = kargs.get('K')
        self.usize=kargs.get("usize")
        self.bsize = kargs.get("bsize")
        self.catsize = kargs.get("catsize")
        self.ub_dict = dict()
        self.bu_dict = dict()
        self.bcat_dict = dict()
        self.catb_dict = dict()

        #self.um_list = list()
        self.user_embedding = np.zeros((self.usize, 128))
        self.item_embedding = np.zeros((self.bsize, 128))
        self.cat_embedding = np.zeros((self.catsize, 128))
        print('Begin to load data')
        start = time.time()
        # HIN2Vec初始化之后用户的嵌入向量表示
        self.load_user_embedding('../data/Toys_and_Games_5.user_embedding')
        self.load_item_embedding('../data/Toys_and_Games_5.item_embedding')
        self.load_cat_embedding('../data/Toys_and_Games_5.cat_embedding')


        self.load_ub(kargs.get('ubfile'))
        self.load_bcat(kargs.get('bcatfile'))
        end = time.time()
        print('Load data finished, used time %.2fs' % (end - start))
        self.path_list = list()
        self.outfile = open(kargs.get('outfile_name'), 'w')
        self.metapath_based_randomwalk()
        self.outfile.close()

    def load_user_embedding(self, ufile):
        with open(ufile) as infile:
            for line in infile.readlines():
                arr = line.strip().split(' ')
                i = int(arr[0])
                for j in range(len(arr[1:])):
                    self.user_embedding[i][j] = float(arr[j + 1])

    def load_item_embedding(self, ifile):
        with open(ifile) as infile:
            for line in infile.readlines():
                arr = line.strip().split(' ')
                i = int(arr[0])
                for j in range(len(arr[1:])):
                    self.item_embedding[i][j] = float(arr[j + 1])
        
    def load_cat_embedding(self, tfile):
        with open(tfile) as infile:
            for line in infile.readlines():
                arr = line.strip().split(' ')
                i = int(arr[0])
                for j in range(len(arr[1:])):
                    self.cat_embedding[i][j] = float(arr[j + 1])
    def metapath_based_randomwalk(self):
        pair_list = []
        for u in range(1, self.usize):
            for i in range(1, self.bsize):
                pair_list.append([u, i])
        print('load pairs finished num = ', len(pair_list))
        ctn = 0
        t1 = time.time()
        avg = 0
        for u, m in pair_list:
            ctn += 1
            #print u, m
            if ctn % 10000 == 0:
                print('%d [%.4f]\n' % (ctn, time.time() - t1))
            if self.metapath == 'ubub':
                path = self.walk_ubub(u, m)
            elif self.metapath == 'ubcb':
                path = self.walk_umtm(u, m)
            else:
                print('unknow metapath.')
                exit(0)
    
    def get_sim(self, u, v):
        return u.dot(v) / ((u.dot(u) ** 0.5) * (v.dot(v) ** 0.5))

    def walk_ubub(self, s_u, e_m):
        limit = 10
        m_list = []
        for m in self.ub_dict[s_u]:
            sim = self.get_sim(self.user_embedding[s_u], self.item_embedding[m])#self.user_embedding[s_u].dot(self.item_embedding[m]) / 
            m_list.append([m, sim])
        m_list.sort(key = lambda x:x[1], reverse = True)
        m_list = m_list[:min(limit, len(m_list))]
        
        u_list = []
        for u in self.bu_dict.get(e_m, []):
            sim = self.get_sim(self.item_embedding[e_m], self.user_embedding[u])#self.item_embedding[e_m].dot(self.user_embedding[u])
            u_list.append([u, sim])
        u_list.sort(key = lambda x:x[1], reverse = True)
        u_list = u_list[:min(limit, len(u_list))]

        mu_list = []
        for m in m_list:
            for u in u_list:
                mm = m[0]
                uu = u[0]
                if mm in self.bu_dict and uu in self.bu_dict[mm] and uu != s_u and mm != e_m:
                    sim = (self.get_sim(self.user_embedding[uu], self.item_embedding[mm]) + u[1] + m[1]) / 3.0
                    if sim > 0.7:
                        mu_list.append([mm, uu, sim])
        mu_list.sort(key = lambda x:x[2], reverse = True)
        mu_list = mu_list[:min(5, len(mu_list))]
        
        if(len(mu_list) == 0):
            return 
        self.outfile.write(str(s_u) + ',' + str(e_m) + '\t' + str(len(mu_list)))
        for mu in mu_list:
            path = ['u' + str(s_u), 'b' + str(mu[0]), 'u' + str(mu[1]), 'b' + str(e_m)]
            self.outfile.write('\t' + '-'.join(path) + ' ' + str(mu[2]))
        self.outfile.write('\n')   
        
    def walk_ubcatb(self,s_u, e_m):
        limit = 10
        m_list = []
        for m in self.ub_dict[s_u]:
            sim = self.get_sim(self.user_embedding[s_u], self.item_embedding[m]) 
            m_list.append([m, sim])
        m_list.sort(key = lambda x:x[1], reverse = True)
        m_list = m_list[:min(limit, len(m_list))]
        
        t_list = []
        for t in self.bcat_dict.get(e_m, []):
            t_list.append([t, 1])

        mt_list = []
        for m in m_list:
            for t in t_list:
                mm = m[0]
                tt = t[0]
                if mm in self.bcat_dict and tt in self.bcat_dict[mm] and  mm != e_m:
                    sim = m[1]
                    if sim > 0.7:
                        mt_list.append([mm, tt, sim])
        mt_list.sort(key = lambda x:x[2], reverse = True)
        mt_list = mt_list[:min(5, len(mt_list))]
        
        if(len(mt_list) == 0):
            return 
        self.outfile.write(str(s_u) + ',' + str(e_m) + '\t' + str(len(mt_list)))
        for mt in mt_list:
            path = ['u' + str(s_u), 'b' + str(mt[0]), 'c' + str(mt[1]), 'b' + str(e_m)]
            self.outfile.write('\t' + '-'.join(path))
        self.outfile.write('\n')   


    def random_walk(self, start):
        path = [self.metapath[0] + start]
        iterator = 0
        k = 1
        while True:
            if k == len(self.metapath):
                iterator += 1
                k = 0
                if iterator == K:
                    return '-'.join(path)

            if k == 0 and self.metapath[k] == 'u':
                pre = path[-1][1:]
                neighbors = self.bu_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1

            elif k == 0 and self.metapath[k] == 'b':
                pre = path[-1][1:]
                neighbors = self.ub_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1
             
            elif self.metapath[k-1] == 'u' and self.metapath[k] == 'b':
                pre = path[-1][1:]
                neighbors = self.ub_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1
            
            elif self.metapath[k-1] == 'b' and self.metapath[k] == 'c':
                pre = path[-1][1:]
                neighbors = self.bcat_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1

            elif self.metapath[k-1] == 'c' and self.metapath[k] == 'b':
                pre = path[-1][1:]
                neighbors = self.catb_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1
                
            elif self.metapath[k-1] == 'b' and self.metapath[k] == 'u':
                pre = path[-1][1:]
                neighbors = self.bu_dict.get(pre, [])
                if len(neighbors) == 0: return None
                index = random.randint(0, len(neighbors) - 1)
                path.append(self.metapath[k] + neighbors[index])
                k += 1



    def load_ub(self, umfile):
        with open(umfile) as infile:
            for line in infile.readlines():
                u, m = line.strip().split('\t')[:2]
                u, m = int(u), int(m)
                #self.um_list.append([u, m]);
                if u not in self.ub_dict:
                    self.ub_dict[u] = list()
                self.ub_dict[u].append(m)

                if m not in self.bu_dict:
                    self.bu_dict[m] = list()
                self.bu_dict[m].append(u)


    def load_bcat(self, mtfile):
        with open(mtfile) as infile:
            for line in infile.readlines():
                m, t= line.strip().split('\t')[:2]
                m, t = int(m), int(t)
                if m not in self.bcat_dict:
                    self.bcat_dict[m] = list()
                self.bcat_dict[m].append(t)

                if t not in self.catb_dict:
                    self.catb_dict[t] = list()
                self.catb_dict[t].append(m)

if __name__ == '__main__':
    ubfile = '../data/train_data.csv'
    bcatfile = '../data/Toys_and_Games_5.bcat' 
    # walk_num = 5
    # metapath = 'ubcatb'

    args = parse_args()
    walk_num = args.walk_num
    metapath = args.metapath

    usize=args.usize

    bsize=args.bsize

    catsize=args.catsize
    K = 1

    # print ("walk_num : ", walk_num, "T : ", type(walk_num))
    # print ("meta : ", metapath, "T : ", type(metapath))
    outfile_name = '../data/Toys_and_Games_5.' + metapath + '_' + str(walk_num) + '_' + str(K)
    print('outfile name = ', outfile_name)
    MetapathBasePathSample(ubfile = ubfile, bcatfile = bcatfile,usize=usize,bsize=bsize,catsize=catsize,
                           K = K, walk_num = walk_num, metapath = metapath, outfile_name = outfile_name)
