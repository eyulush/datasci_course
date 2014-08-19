.output union.txt
SELECT COUNT(DISTINCT term) FROM (
	SELECT * FROM Frequency WHERE docid = '10398_txt_earn' OR docid = '925_txt_trade'
) WHERE count = 1;
.output stdout