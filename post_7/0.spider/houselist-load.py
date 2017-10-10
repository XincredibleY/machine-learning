
# coding: utf-8

# In[ ]:


# load the house list under some city and save it to db
import urllib
import json
from pymongo import MongoClient
import time
import sys
# from bson import ObjectId
# connect to mongo
conn = MongoClient('localhost',27017)
# use house db
db = conn.house


# convert string to json
def str2json(str):
    return json.loads(str)

# save the list to db
def saveListToDb(json,city_code):
    houselist = db.houselist
    result = houselist.insert_one(json)
    # add insert time 
    insert_time = int(time.time())
    houselist.update_one({'_id':result.inserted_id},{"$set":{"insert_time":insert_time,"city_code":city_code}})
    
# check room id if exist , return True
def check_room_id_exist(room_id):
    houselist = db.houselist
    if houselist.count({'room_id':room_id}) > 0:
        return True
    else:
        return False
    
# load house list
def load_house_list(city_name,page):
    num_index = 1
    
    city_code_dict = {
        'shanghai':'001009001',
        'beijing':'001001',
        'hangzhou':'001011001',
        'shenzheng':'001019002',
        'nanjing':'001010001'
    }
    print "city:%s \n\n" % city_name
    city_code = city_code_dict[city_name]
    while True:
        list_url= 'http://m.hizhu.com/Home/House/houselist.html?city_code='+city_code+'&pageno='+str(page)+'&limit=10&sort=-1&region_id=&plate_id=&money_max=999999&money_min=0&logicSort=0&line_id=0&stand_id=0&key=&key_self=0&type_no=0&search_id=&latitude=&longitude=&distance=0&update_time=1502327960'
        json_data = json.loads(urllib.urlopen(list_url).read()) 
        # check data.house_list is null
        if json_data['data']['house_list'] is None:
            print 'all page done'
            break
        else:
            for house_item  in json_data['data']['house_list']:
                # check house existence
                if not check_room_id_exist(house_item['room_id']):
                    saveListToDb(house_item,city_code)
                    print 'num % d :room id  %s saved ' % (num_index,house_item['room_id'])
                    num_index+=1
                else:
                    print 'room id  %s existed ' % house_item['room_id']
            print 'page %d done \n' % page
        page+=1

city_name = sys.argv[1]
page = int(sys.argv[2])

load_house_list(city_name,page)
