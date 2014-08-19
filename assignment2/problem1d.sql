.output count.txt
SELECT count(DISTINCT docid) FROM Frequency where term = 'parliament';
.output stdout