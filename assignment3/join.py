import sys
import json
import MapReduce

mr = MapReduce.MapReduce()

def mapper(record):
	# key : order id
	# value : order or iterm attribute
	key = record[1]
	mr.emit_intermediate(key,record)
	
def reducer(key, list_of_values):
	# key : order id
	# value : each value is an order or iterm record	
	order = list_of_values[0]
	for v in list_of_values[1:len(list_of_values)]:
		mr.emit(order+v)

inputdata = open(sys.argv[1])
mr.execute(inputdata,mapper,reducer)
			
