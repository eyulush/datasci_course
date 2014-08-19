import sys
import json
import MapReduce

mr = MapReduce.MapReduce()

def mapper(record):
	# key : document identifier
	# value : document contents
	key = record[0]
	value = record[1]
	words = value.split()
	for w in words:
		mr.emit_intermediate(w,key)
	
def reducer(key, list_of_values):
	# key : word
	# value : list of document identifier	
	new_list_of_values = []
	for v in list_of_values:
		if v not in new_list_of_values:
			new_list_of_values.append(v)
	
	record = (key,new_list_of_values)
	
	mr.emit(record)

inputdata = open(sys.argv[1])
mr.execute(inputdata,mapper,reducer)
			
