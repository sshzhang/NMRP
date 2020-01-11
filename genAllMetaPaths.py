import sys
import os
import random
from collections import Counter


class MetaPathGenerator:
    def __init__(self):
        # 保存 id 用户
        self.id_user = dict()
        # 保存id 商品
        self.id_item = dict()
        # 保存id 类别
        self.id_cat = dict()

        self.user_item = dict()
        self.item_user = dict()
        self.item_cat = dict()
        self.cat_item = dict()

    def read_data(self, dirpath):
        with open(dirpath + "/id_u.txt") as adictfile:
            for line in adictfile:
                toks = line.strip().split(" ")
                if len(toks) == 2:
                    self.id_user[toks[0]] = toks[1].replace(" ", "")

        # print "#users", len(self.id_user)

        with open(dirpath + "/id_item.txt") as cdictfile:
            for line in cdictfile:
                toks = line.strip().split(" ")
                if len(toks) == 2:
                    newconf = toks[1].replace(" ", "")
                    self.id_item[toks[0]] = newconf

        with open(dirpath + "/id_ca.txt") as cdictfile:
            for line in cdictfile:
                toks = line.strip().split("--")
                if len(toks) == 2:
                    # newconf = toks[1].replace(" ", "")
                    self.id_cat[toks[0]] = toks[1]

        # print "#cat", len(self.id_ca)

        with open(dirpath + "/user_item.txt") as pafile:
            for line in pafile:
                toks = line.strip().split(" ")
                if len(toks) == 3:
                    p, a = toks[0], toks[1]
                    if p not in self.user_item:
                        self.user_item[p] = []
                    self.user_item[p].append(a)
                    if a not in self.item_user:
                        self.item_user[a] = []
                    self.item_user[a].append(p)

        with open(dirpath + "/item_cat.txt") as pcfile:
            for line in pcfile:
                toks = line.strip().split(" ")
                if len(toks) == 2:
                    p, c = toks[0], toks[1]
                    self.item_cat[p] = c
                    if c not in self.cat_item:
                        self.cat_item[c] = []
                    self.cat_item[c].append(p)

    def generate_random_UBCatB(self, outfilename):  # 元路径UBUB生成策略类似
        # 此方法生成UBCatBU,
        outfile = open(outfilename, 'w')
        for u1 in self.user_item:
            # outline=self.id_user[u1]
            for i1 in self.user_item[u1]:
                outline = self.id_user[u1] + " " + self.id_item[i1]

                if i1 not in self.item_cat: continue
                for i in range(0, 500):
                    categories = self.item_cat[i1]
                    numa = len(categories)
                    catid = random.randrange(numa)
                    cat = categories[catid]
                    # outline += " " + self.id_cat[cat]
                    outline += " " + self.id_cat[cat].replace("\'", "").replace(" ", "")
                    items = self.cat_item[cat]
                    numb = len(items)
                    itemid = random.randrange(numb)
                    item = items[itemid]
                    outline += " " + self.id_item[item]

                    users = self.item_user[item]
                    numc = len(users)
                    userid = random.randrange(numc)
                    user = users[userid]
                    outline += " " + self.id_user[user]
                outfile.write(outline + "\n")

        outfile.close()


    def generate_random_UBUB(self, outfilename):

        outfile = open(outfilename, 'w')
        for u1 in self.user_item:
            # outline=self.id_user[u1]
            for i1 in self.user_item[u1]:
                outline = self.id_user[u1] + " " + self.id_item[i1]
                # 如果只有当前用户购买过此商品
                if i1 not in self.item_user or len(self.item_user[i1])==1 : continue

                for i in range(0, 500):
                    users = self.item_user[i1]
                    userc = users.copy()
                    userc.remove(u1)
                    numa = len(userc)
                    userid = random.randrange(numa)
                    user = users[userid]
                    outline += " "+self.id_user[user].replace("\'", "").replace(" ", "")
                    items = self.user_item[user]
                    numb = len(items)
                    itemid = random.randrange(numb)
                    item = items[itemid]
                    outline += " "+self.id_item[item]

                outfile.write(outline + "\n")

        outfile.close()

# for cat in self.item_cat[i1]:
#     for i2 in self.cat_item[cat]:
#         if i2==i1: continue
#         for u2 in self.item_user[i2]:
#             if u2==u1: continue
#             outline += " " + self.id_item[i1]+" "+self.id_cat[cat]+" "+self.id_item[i2]+" "+self.id_user[u2]
#             outfile.write(outline + "\n")


# python genAllMetaPaths.py.py 1000 100 net_aminer output.aminer.w1000.l100.txt
# python genAllMetaPaths.py.py 1000 100 net_dbis   output.dbis.w1000.l100.txt




