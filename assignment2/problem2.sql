.output multiply.txt 
SELECT multiply_value from (
	SELECT a.row_num as row_num, b.col_num as col_num, sum(a.value*b.value) as multiply_value FROM a,b 
	WHERE a.col_num = b.row_num
	GROUP BY a.row_num, b.col_num) 
WHERE row_num = 2 and col_num = 3;
.output stdout