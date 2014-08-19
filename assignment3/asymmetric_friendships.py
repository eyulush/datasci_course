import sys
import json
import MapReduce

mr = MapReduce.MapReduce()

def mapper(record):
	# key : person A
	# value : friend of person A
	key = (record[0],record[1])
	mr.emit_intermediate(key,1)
	key = (record[1],record[0])
	mr.emit_intermediate(key,1)
	
def reducer(key, list_of_values):
	# key : person A
	# value : friends of person A	

	if len(list_of_values) == 1:
		mr.emit(key)

inputdata = open(sys.argv[1])
mr.execute(inputdata,mapper,reducer)
			
