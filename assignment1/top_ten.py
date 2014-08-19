import sys
import json
import re


def top_ten_hashtag(tweet_file):      
    hashtag_freq={}
    for line in tweet_file:
        tweet_data = json.loads(line)
        try:
            tweet_hashtags = tweet_data.get('entities').get('hashtags')
            for hashtag in tweet_hashtags:
                hashtag_text = hashtag['text']
                hashtag_freq[hashtag_text] = hashtag_freq.get(hashtag_text,0) + 1
        except:
            continue
    sorted_hashtag_freq = sorted(hashtag_freq.iteritems(), key=lambda x : x[1], reverse=True) 
    counter = 0 
    for hashtag_freq in sorted_hashtag_freq:
        try:
            print "%s %d" %(hashtag_freq[0],hashtag_freq[1])
            counter = counter + 1
            if counter >= 10:
                return
        except:
            continue
                 
    
def main():
    tweet_file = open(sys.argv[1])
    top_ten_hashtag(tweet_file)

if __name__ == '__main__':
    main()