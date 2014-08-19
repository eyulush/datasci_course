.output similarity_matrix.txt 
	SELECT sum(d.count*dt.count) as var_count FROM Frequency d, Frequency dt
	WHERE d.term = dt.term and d.docid = '10080_txt_crude' and dt.docid = '17035_txt_earn'
	GROUP BY d.docid, dt.docid;
.output stdout