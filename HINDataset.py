import scipy.sparse as sp
import numpy as np

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        self.types = {'u' : 1, 'b' : 2, 'c' : 3}  # u表示用户, b表示商品, c表示类别
        self.trainMatrix = self.load_rating_file_as_matrix(path + "train_data.csv")
        self.num_users, self.num_items = self.trainMatrix.shape[0], self.trainMatrix.shape[1]
        
        self.user_item_map, self.item_user_map, self.train, self.item_popularity = self.load_rating_file_as_map(path + "train_data.csv")
        self.testRatings = self.load_rating_file_as_list(path + "test_data.csv")
        self.user_feature, self.item_feature, self.cat_feature = self.load_feature_as_map(path+'user_embedding', path+'item_embedding', path+'cat_embedding')
        self.fea_size = len(self.user_feature[1])
        # self.ubcatb_path_num 此元路径下最多的路劲数, ubcatb_timestamp元路劲节点的个数
        self.path_ubcatb, self.ubcatb_path_num, self.ubcatb_timestamp = self.load_path_as_map(path + "ubcb_5_1")
        self.path_ubub, self.ubub_path_num, self.ubub_timestamp = self.load_path_as_map(path + 'ubub_5_1')

        
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                tmp_list = []
                for i in arr:
                    tmp_list.append(int(i))
                ratingList.append(tmp_list)
                line = f.readline()
        return ratingList

    
    def load_rating_file_as_map(self, filename):
        user_item_map = {}
        item_user_map = {}
        train = []
        popularity_dict = {}
        max_i = 0
        total = 0
        with open(filename) as f:
            line = f.readline()
            while line != None and line != '':
                arr = line.strip().split('\t')
                u, i = int(arr[0]), int(arr[1])
                #self.num_users = max(self.num_users, u)
                max_i = max(max_i, i)
                if u not in user_item_map:
                    user_item_map[u] = {}
                if i not in item_user_map:
                    item_user_map[i] = {}
                if i not in popularity_dict:
                    popularity_dict[i] = 0
                user_item_map[u][i] = 1.0
                item_user_map[i][u] = 1.0
                popularity_dict[i] += 1
                total += 1
                train.append([u, i])
                line = f.readline()
        #self.num_users += 1
        #self.num_items += 1
        item_popularity = [0] * max_i
        for i in popularity_dict:
            item_popularity[i - 1] = int(popularity_dict[i] ** 0.5)
        return user_item_map, item_user_map, train, item_popularity

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        train_list = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                train_list.append([user, item])
                mat[user, item] = 1.0
                line = f.readline()    
        return mat

    def load_feature_as_map(self, user_fea_file, item_fea_file, cat_fea_file):
        user_feature = np.zeros((self.num_users, 128))
        item_feature = np.zeros((self.num_items, 128))
        cat_feature = np.zeros((19, 128))

        with open(user_fea_file) as infile:
            for line in infile.readlines():
                arr = line.strip().split(' ')
                u = int(arr[0])
                #user_feature[u] = list()
                for j in range(len(arr[1:])):
                    user_feature[u][j] = float(arr[j + 1])

        with open(item_fea_file) as infile:
            for line in infile.readlines():
                arr = line.strip().split(' ')
                i = int(arr[0])
                #item_feature[i] = list()
                for j in range(len(arr[1:])):
                    item_feature[i][j] = float(arr[j + 1])

        with open(cat_fea_file) as infile:
            for line in infile.readlines():
                arr = line.strip().split(' ')
                t = int(arr[0])
                #cat_feature[t] = list()
                for j in range(len(arr[1:])):
                    cat_feature[t][j] = float(arr[j + 1])

        return user_feature, item_feature, cat_feature
    
    def load_path_as_map(self, filename):  # 1,12	3	u1-b56-c6-b12	u1-b79-c16-b12	u1-b195-c16-b12
        print(filename)
        path_dict = {}
        path_num = 0
        timestamps = 0
        length = 2
        ctn = 0
        with open(filename) as infile:
            line = infile.readline()
            while line != None and line != "":
                arr = line.split('\t')
                u, i = arr[0].split(',')  # 1,12
                u = int(u)
                i = int(i) 
                path_dict[(u, i)] = []
                path_num = max(int(arr[1]), path_num)
                # 路径中节点个数
                timestamps = len(arr[2].strip().split('-'))
                line = infile.readline()
                # 行数
                ctn += 1
        print(ctn, path_num, timestamps, length)
        with open(filename) as infile:
            line = infile.readline()
            while line != None and line != "":
                arr = line.strip().split('\t')
                u, i = arr[0].split(',')
                u, i = int(u), int(i)
                path_dict[(u, i)] = []
            
                for path in arr[2:]:
                    tmp = path.split(' ')[0].split('-')
                    node_list = []  # [[数据类型, 对应索引号],[]]
                    for node in tmp:
                        # 相应的索引号
                        index = int(node[1:])
                        node_list.append([self.types[node[0]], index])
                    path_dict[(u, i)].append(node_list)
                line = infile.readline()
        return path_dict, path_num, timestamps


if __name__ == '__main__':
    dataset = Dataset('./data/')
    print( dataset.user_feature)
    print( dataset.item_feature)
    print( dataset.cat_feature)
