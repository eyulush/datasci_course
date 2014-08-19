import sys
import json
import re

def hw():
    print 'Hello, world!'

def lines(fp):
    print str(len(fp.readlines()))

def inc_term_count(term, counter):
    if term in counter:
        counter[term] = counter[term] + 1
    else:
        counter[term] = 1
    return counter

def inc_conn_count(term1,term2,term_conn_counter):
    if term_conn_counter.get((term1,term2),0) == 0:
        term_conn_counter[(term1,term2)] = 1
    else:
        term_conn_counter[(term1,term2)] = term_conn_counter[(term1,term2)] + 1
    return term_conn_counter


def determine_score(sent_file, tweet_file):
    sent_term_scores = {} # initialize an empty dictionary
    non_sent_term_scores = {} # initialize an empty dictionary
    
    # Load the scores
    for line in sent_file:
        term, score  = line.split("\t")  # The file is tab-delimited. "\t" means "tab character"
        sent_term_scores[term] = int(score)  # Convert the score to an integer.
        
    non_sent_term_count = {}
    sent_term_count = {}
    term_conn_count = {}
    
    for line in tweet_file:
        tweet_data = json.loads(line)
        tweet_text = tweet_data.get('text')
        if type(tweet_text) is unicode and tweet_data.get('lang','non') == 'en':
            tweet_words = re.split(r'[\s\,\.\?\:\"\-\@\;\']+',tweet_text)        
            tweet_words = filter(lambda word: word.isalpha(), tweet_words)  # filter out non alnum string
            tweet_words = map(lambda word: word.lower(), tweet_words)       # trans to lowercase
            non_sent_terms = [] # initialize the list
            sent_terms = []     # initialize the list
            for word in tweet_words:
                if word in sent_term_scores:
                    sent_terms.append(word)
                else:
                    non_sent_terms.append(word)
            for non_sent_term in non_sent_terms:
                non_sent_term_count = inc_term_count(non_sent_term,non_sent_term_count)
                for sent_term in sent_terms:
                    term_conn_count = inc_conn_count(non_sent_term,sent_term,term_conn_count)
            for sent_term in sent_terms:
                sent_term_count = inc_term_count(sent_term,sent_term_count)  
                                   
    for nst_term, nst_count in non_sent_term_count.iteritems():
        nst_score = 0.0
        for st_term,st_count in sent_term_count.iteritems():
            conn_count = term_conn_count.get((nst_term,st_term),0)
            if conn_count > 0:
                conn_weight = float((conn_count*conn_count))/float((nst_count*st_count))
                nst_score = nst_score + conn_weight*sent_term_scores[st_term]  
        non_sent_term_scores[nst_term] = nst_score

    for nst_term, nst_score in non_sent_term_scores.iteritems():
        try:
            print '%s %f' %(nst_term, nst_score)
        except:
            continue
    
    #for st_term, st_score in sent_term_scores.iteritems():
    #    print '%s %f' %(st_term, st_score)
    
          
    
def main():
    sent_file = open(sys.argv[1])
    tweet_file = open(sys.argv[2])
    #hw()
    #lines(sent_file)
    #lines(tweet_file)
    determine_score(sent_file,tweet_file)
    sent_file.close()
    tweet_file.close()
    

if __name__ == '__main__':
    main()
