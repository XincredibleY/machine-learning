
# coding: utf-8

# In[1]:


import urllib
import pymongo 
import time
import json
import sys


# In[ ]:

# connect mongo
conn = pymongo.MongoClient('localhost',27017)
# use house 
db = conn.house
# collection houselist
houselist = db.houselist
# collection housedetail
housedetail = db.housedetail


# In[ ]:

# get house detail by room id
def getHouseDetail(room_id):
    detail_url = 'http://m.hizhu.com/Home/House/housedetail.html?room_id='+room_id+'&city_code=001011001'
    json_data = json.loads(urllib.urlopen(detail_url).read())
    return json_data

# save json data to mongo db
def save2db(json_data):
    # insert data
    result = housedetail.insert_one(json_data)
    # update unix time str
    now_time=  int(time.time())
    housedetail.update_one({'_id':result.inserted_id},{"$set":{"insert_time":now_time}})

    
def check_room_id_exist(room_id):
    if housedetail.count({'room_id':room_id}) > 0:
        return True
    else:
        return False    

# get data from mongo
def load_data(city_code,skip_num,limit_num):
	return houselist.find({"city_code":city_code}).sort("insert_time",pymongo.DESCENDING).skip(skip_num).limit(limit_num)
# get list from houselist collection and save it 
def main(city_name,skip_num):
    
    city_code_dict = {
        'shanghai':'001009001',
        'beijing':'001001',
        'hangzhou':'001011001',
        'shenzheng':'001019002',
        'nanjing':'001010001'
    }

    city_code = city_code_dict[city_name]
    index_num = 1


    limit_num = 100
    
    
    data_count = houselist.count({"city_code":city_code})
    pages_num = (data_count / limit_num)+1

    for page_index in range(pages_num):
    	houseListData = load_data(city_code,skip_num,limit_num)
    	skip_num = page_index * limit_num
    	for houseItem in houseListData:
    		room_id =  houseItem['room_id']
    		if not check_room_id_exist(room_id):

    			json_data = getHouseDetail(room_id)
    			if json_data['status'] == '200':
    				save2db(json_data['data'])
                		print 'num %d : room id: %s saved' % (index_num,room_id)
                		index_num +=1
                	else:
                		print 'num %d : room id: %s error loaded' % (index_num,room_id)
		else:
    			print 'room id: %s existed' % (room_id)
    	
    
# load room id 
def load_room_id(skip_num,limit_num,sort_index):
	# get all room id in housedetail
	housedetail_room_ids = []
	for item in  housedetail.find({},{"room_id":1}):
		housedetail_room_ids.append(item['room_id'])
	# then get room id in housedetail that doesn't locate in houselist
	real_room_ids = houselist.find({"room_id":{"$nin":housedetail_room_ids}},{"room_id":1}).skip(skip_num).limit(limit_num).sort("insert_time",(pymongo.DESCENDING if sort_index =="desc" else pymongo.ASCENDING))
	room_ids_return = []
	for i in real_room_ids:
		room_ids_return.append(i['room_id'])
	print 'all room id num is %d \n\n' % len(room_ids_return)
	return room_ids_return	


# better way to check 
def main_better(skip_num,limit_num,sort_index):
	index_num = 1
	for room_id in load_room_id(skip_num,limit_num,sort_index):

		if not check_room_id_exist(room_id):

    			json_data = getHouseDetail(room_id)
    			if json_data['status'] == '200':
    				save2db(json_data['data'])
                		print 'num %d : room id: %s saved' % (index_num,room_id)
                		index_num +=1
                	else:
                		print 'num %d : room id: %s error loaded' % (index_num,room_id)
		else:
    			print 'room id: %s existed' % (room_id)
    	print '\n\n all done'


# In[ ]:

skip_num,limit_num,sort_index = int(sys.argv[1]),int(sys.argv[2]),sys.argv[3]
main_better(skip_num,limit_num,sort_index)



# print get_room_ids(0,10)

# In[ ]:



