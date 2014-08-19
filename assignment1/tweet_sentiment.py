import sys
import json
import re

def hw():
    print 'Hello, world!'

def lines(fp):
    print str(len(fp.readlines()))

def calcSentScore(sent_file, tweet_file):
    scores = {} # initialize an empty dictionary
    # Load the scores
    for line in sent_file:
        term, score  = line.split("\t")  # The file is tab-delimited. "\t" means "tab character"
        scores[term] = int(score)  # Convert the score to an integer.
    # print scores.items() # Print every (term, score) pair in the dictionary
    # Load the twitter data
    for line in tweet_file:
        tweet_data = json.loads(line)
        tweet_text = tweet_data.get('text')
        #print tweet_data.keys();
        score = 0; # initialize the score as 0
        if type(tweet_text) is unicode:
            tweet_words = re.split(r'[\s\,\.\?\:\"\-]+',tweet_text)        
            for word in tweet_words:
                score = score + scores.get(word,0)
        print score
    
def main():
    sent_file = open(sys.argv[1])
    tweet_file = open(sys.argv[2])
    #hw()
    #lines(sent_file)
    #lines(tweet_file)
    # Calculate the sentiment for each twitter
    calcSentScore(sent_file, tweet_file)
    

if __name__ == '__main__':
    main()
