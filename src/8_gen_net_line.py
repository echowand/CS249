import itertools

def gen_displayid_uuid():
	inputfile = "/Users/DavidZhou/GDriveUCLA/Study/Win17/CS249-Sun/project/data/events_small.csv"
	fin = open(inputfile)
	u_dict = {}

	i = -1
	for l in fin:
		i=i+1
		if i == 0:
			continue
		l = l.split(',')
		uuid = l[1]
		displayid = l[0]
		# print uuid, displayid
		if uuid not in u_dict:
			u_dict[uuid]=[displayid]
		else:
			u_dict[uuid].append(displayid)
	# print u_dict
	for k in u_dict:
		mylist = u_dict[k]
		if len(mylist) > 1:
			for item in itertools.combinations(mylist,2):
				print item[0], item[1]

	# break

gen_displayid_uuid()	

