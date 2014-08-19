.output two_words.txt 
SELECT count(DISTINCT docid) FROM (SELECT * From Frequency WHERE docid IN (SELECT docid FROM Frequency where term = 'transaction')) WHERE term = 'world';
.output stdout