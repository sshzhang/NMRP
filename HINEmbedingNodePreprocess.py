import  json
import  re
import  os

Path="."
def GenarateTheidUser():

    with open('./Toys_and_Games_5.json', 'r') as fr:
        user_map={}
        r_user_map={}
        item_map={}
        r_item_map={}
        user_item_rating={}
        user_item_review={}
        lines = fr.readlines()
        for line in lines:
            linejs = json.loads(line)
            u_id=linejs['reviewerID']
            r_id=linejs['asin']
            rating=linejs['overall']
            reviewText=linejs['reviewText']

            if u_id not in user_map.values():
                user_map[len(user_map.keys())] = u_id
                r_user_map[u_id] = len(user_map.keys())-1

            if r_id not in item_map.values():
                item_map[len(item_map.keys())] = r_id
                r_item_map[r_id] = len(item_map.keys())-1

            if (u_id, r_id) not in user_item_rating.keys():
                user_item_rating[(u_id, r_id)] = rating

            if (u_id, r_id) not in user_item_review.keys():
                user_item_review[(u_id, r_id)] = reviewText

    print("start writing id_u.txt")

    with open(os.path.join(Path, 'id_u.txt'), 'w+') as fm:
        for u_id in user_map.keys():
            reviewer_id = user_map[u_id]
            fm.write("%d %s\n"%(int(u_id), reviewer_id))

    print("start writing id_item.txt")

    with open(os.path.join(Path, "id_item.txt"), 'w+') as fm:
        for r_id in item_map.keys():
            item_id=item_map[r_id]
            fm.write("%d %s\n"%(r_id, item_id))

    print("start writing u_item.txt")
    with open(os.path.join(Path, 'u_i.txt'), 'w+') as ui:
        for (u_id, r_id) in user_item_rating.keys():
            rating = user_item_rating[(u_id, r_id)]
            ui.write("%s %s %s\n"%(u_id, r_id, rating))

    with open(os.path.join(Path, "user_item.txt"), 'w+') as uiid:
        for (u_id, r_id) in user_item_rating.keys():
            rating = user_item_rating[(u_id, r_id)]
            uiid.write("%s %s %s\n"%(r_user_map[u_id], r_item_map[r_id], rating))


    categoried={}
    r_categoried={}
    item_cat={}

    brands={}
    r_brands={}
    item_brands={}


    with open("meta_Toys_and_Games.json", 'r') as ftm:
        lines = ftm.readlines()
        for line in lines:
            result1 = re.search("\'asin\': \'(?P<asign>.*?)\', .*\'categories\': \[\[(?P<cat>.*)\]\]", line)
            asign = result1.group('asign')
            cat = result1.group('cat')

            results = re.search("\'brand\': \'(?P<brand>.*?)\'", line)
            if results is not None and results is not '\n':
                brand = results.group('brand').strip()
                print(brand)
                if asign in item_map.values():
                    bs=brand.split(";")
                    for b in bs:
                        if b not in brands.values():
                            brands[len(brands.keys())]=b
                            r_brands[b]=len(brands.keys())-1
                            if r_item_map[asign] not in item_brands.keys():
                                item_brands[r_item_map[asign]]=[r_brands[b]]
                            else: item_brands[r_item_map[asign]].append(r_brands[b])




            if cat.startswith("["):
                cat = cat[1:]
            if cat.endswith("]"):
                cat = cat[:len(ca)-1]
            cat = cat.replace('[', '')
            cat = cat.replace(']', '')
            print(asign)
            print(cat)
            if asign in item_map.values():
                cats=cat.split(',')
                for ca in cats:
                    if ca not in categoried.values():
                        categoried[len(categoried.keys())] = ca
                        r_categoried[ca]=len(categoried.keys())-1
                    if r_item_map[asign] not in item_cat.keys():
                        item_cat[r_item_map[asign]]=[r_categoried[ca]]
                    else: item_cat[r_item_map[asign]].append(r_categoried[ca])




    print("start writing id_ca.txt")
    with open(os.path.join(Path, 'id_ca.txt'), 'w+') as ica:
        for cid in categoried.keys():
            cate_id=categoried[cid]
            ica.write("%d--%s\n"%(cid,cate_id))

    print("start writing id_brd.txt")
    with open(os.path.join(Path, 'id_brd.txt'), 'w+') as ica:
        for cid in brands.keys():
            cate_id = brands[cid]
            ica.write("%d--%s\n" % (cid, cate_id))


    print("start writing item_cat.txt")
    with open(os.path.join(Path, 'item_cat.txt'),'w+') as itca:
        for item_id in item_cat.keys():
             for cat_id in item_cat[item_id]:
                 itca.write("%d %s\n"%(item_id,cat_id))

    print("start writing item_brand.txt")
    with open(os.path.join(Path, 'item_brand.txt'), 'w+') as itca:
        for item_id in item_brands.keys():
            for brd_id in item_brands[item_id]:
                itca.write("%d %s\n" % (item_id, brd_id))



# GenarateTheidUser()

# with open("meta_Toys_and_Games.json", 'r') as ftm:
#     lines = ftm.readlines()
#     for line in lines:
#         results = re.search("\'brand\': \'(?P<brand>.*?)\'", line)
#         if results is not None and results is not '\n':
#             brand = results.group('brand').strip()
#             print(brand)


with open(os.path.join(Path, "user_item.txt"), 'r') as ftm:
    lines = ftm.readlines()
    print(len(lines))


