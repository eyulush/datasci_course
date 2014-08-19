.output keyword_search.txt
SELECT similiarity_count FROM (
	SELECT freq.docid, sum(freq.count*query.count) as similiarity_count FROM Frequency freq, (
		SELECT 'q' as docid, 'washington' as term, 1 as count 
		UNION
		SELECT 'q' as docid, 'taxes' as term, 1 as count
		UNION 
		SELECT 'q' as docid, 'treasury' as term, 1 as count) as query
	WHERE freq.term = query.term 
	GROUP BY freq.docid
	ORDER BY similiarity_count DESC)
LIMIT 1;
.output stdout