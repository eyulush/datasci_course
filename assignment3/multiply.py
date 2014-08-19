import sys
import json
import MapReduce

mr = MapReduce.MapReduce()

# Without information of Matrix A and Matrix B Dimensions
# it is hard to do this matrix multiply with single MapReducer
# see the discussion https://class.coursera.org/datasci-002/forum/thread?thread_id=1436

# predefine matrix dimension, in fact, we could pre-scan data to get dimensions.
# here, we simply define a variable for dimension of result matrix
mat_dimension = (5,5)

def mapper(record):	
	# key :  (i,j) for result matrix
	# value : (k,v) for k-th value
	matrix_name = record[0]
	if matrix_name == "a":
		i = record[1]
		for j in range(mat_dimension[1]):
			key = (i,j)
			value = (record[2],record[3])
			mr.emit_intermediate(key,value)
	if matrix_name == "b":
		j = record[2]
		for i in range(mat_dimension[0]):
			key = (i,j)
			value = (record[1],record[3])
			mr.emit_intermediate(key,value)
	
def reducer(key, list_of_values):
	# key : (i,j) for result matrix
	# list_of_values : k-th value
	# find value pairs for k-th
	value_dict = {}
	for v in list_of_values:
		value_dict.setdefault(v[0], [])
		value_dict[v[0]].append(v[1])

	# sum the multiply result for the value at (i,j) in result matrix
	result = 0
	for kv in value_dict:
		value = value_dict[kv]
		if len(value) == 2:
			result = result + value[0]*value[1]
			
	if result > 0:
		mr.emit(key + (result,))
	
inputdata = open(sys.argv[1])
mr.execute(inputdata,mapper,reducer)
			
