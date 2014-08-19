.output big_documents.txt 
SELECT count(DISTINCT docid) FROM Frequency WHERE docid IN (SELECT docid FROM Frequency GROUP BY docid HAVING SUM(count) > 300);
.output stdout