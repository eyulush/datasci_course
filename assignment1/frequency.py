import sys
import json
import re

def calcFrequency(tweet_file):
    term_occurs = {} # initialize an empty dictionary
    total_term_occurs = 0;
    for line in tweet_file:
        tweet_data = json.loads(line)
        tweet_text = tweet_data.get('text')
        if type(tweet_text) is unicode:
            tweet_words = re.split(r'[\s\,\.\?\:\"\-\@\/\']+',tweet_text)        
            for term in tweet_words:
                if term.isalnum():
                    term_occurs[term] = term_occurs.get(term,0.0) + 1.0
                    total_term_occurs = total_term_occurs + 1.0
    # print term frequency
    for key in term_occurs:
        print '%s %f' %(key, term_occurs[key] / total_term_occurs)
    
def main():
    tweet_file = open(sys.argv[1])
    calcFrequency(tweet_file)

if __name__ == '__main__':
    main()