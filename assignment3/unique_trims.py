import sys
import json
import MapReduce

mr = MapReduce.MapReduce()

def mapper(record):
	# key : sequence id
	# value : nucleotides
	seq_id = record[0]
	nucleotides = record[1]
	mr.emit_intermediate(nucleotides[0:len(nucleotides)-10],seq_id)
	
def reducer(key, list_of_values):
	# key : trimmed nucleotides
	# value : seq_id	
	mr.emit(key)

inputdata = open(sys.argv[1])
mr.execute(inputdata,mapper,reducer)
			
