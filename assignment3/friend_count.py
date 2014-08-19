import sys
import json
import MapReduce

mr = MapReduce.MapReduce()

def mapper(record):
	# key : person A
	# value : friend of person A
	key = record[0]
	mr.emit_intermediate(key,record[1])
	
def reducer(key, list_of_values):
	# key : person A
	# value : friends of person A		
	mr.emit((key,len(list_of_values)))

inputdata = open(sys.argv[1])
mr.execute(inputdata,mapper,reducer)
			
