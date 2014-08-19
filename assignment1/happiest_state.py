import sys
import json
import re

State_Abbr_Name = {'Alabama'          :  'AL', 
                   'Alaska'           :  'AK', 
                   'Arizona'          :  'AZ', 
                   'Arkansas'         :  'AR', 
                   'California'       :  'CA', 
                   'Colorado'         :  'CO', 
                   'Connecticut'      :  'CT', 
                   'Delaware'         :  'DE', 
                   'Florida'          :  'FL', 
                   'Georgia'          :  'GA', 
                   'Hawaii'           :  'HI', 
                   'Idaho'            :  'ID', 
                   'Illinois'         :  'IL', 
                   'Indiana'          :  'IN', 
                   'Iowa'             :  'IA', 
                   'Kansas'           :  'KS', 
                   'Kentucky'         :  'KY', 
                   'Louisiana'        :  'LA', 
                   'Maine'            :  'ME', 
                   'Maryland'         :  'MD', 
                   'Massachusetts'    :  'MA', 
                   'Michigan'         :  'MI', 
                   'Minnesota'        :  'MN', 
                   'Mississippi'      :  'MS', 
                   'Missouri'         :  'MO', 
                   'Montana'          :  'MT', 
                   'Nebraska'         :  'NE', 
                   'Nevada'           :  'NV', 
                   'New Hampshire'    :  'NH', 
                   'New Jersey'       :  'NJ', 
                   'New Mexico'       :  'NM', 
                   'New York'         :  'NY', 
                   'North Carolina'   :  'NC', 
                   'North Dakota'     :  'ND', 
                   'Ohio'             :  'OH', 
                   'Oklahoma'         :  'OK', 
                   'Oregon'           :  'OR', 
                   'Pennsylvania'     :  'PA', 
                   'Rhode Island'     :  'RI', 
                   'South Carolina'   :  'SC', 
                   'South Dakota'     :  'SD', 
                   'Tennessee'        :  'TN', 
                   'Texas'            :  'TX', 
                   'Utah'             :  'UT', 
                   'Vermont'          :  'VT', 
                   'Virginia'         :  'VA', 
                   'Washington'       :  'WA', 
                   'West Virginia'    :  'WV', 
                   'Wisconsin'        :  'WI', 
                   'Wyoming'          :  'WY'}
    
def getState(place_name):
    place = re.split(r',\s+',place_name)
    if place[-1].upper() == 'USA' or place[-1].upper() == 'DC':
        return State_Abbr_Name.get(place[-2])
    if place[-1].upper() in State_Abbr_Name.values():
        return place[-1]
    return 'NotFound'

def getScore(text, term_scores):
    score = 0
    words = re.split(r'[\s\,\.\?\:\"\-]+',text)        
    for word in words:
        score = score + term_scores.get(word,0)
    return score

def happiest_state(sent_file, tweet_file):  
    # Load the scores
    term_scores={}
    for line in sent_file:
        term, score  = line.split("\t")  # The file is tab-delimited. "\t" means "tab character"
        term_scores[term] = int(score)  # Convert the score to an integer.
    
    state_score = {} # initialize an empty dictionary    
    for line in tweet_file:
        tweet_data = json.loads(line)
        tweet_text = tweet_data.get('text')
        if tweet_data.get('place', None) != None and type(tweet_text) is unicode:
            if tweet_data.get('place').get('country_code') == 'US':         
                state = getState(tweet_data.get('place').get('full_name'))
                score = getScore(tweet_text,term_scores)
                state_score[state] = state_score.get(state,0) + score;
    
    highest_score = 0
    happiest_state = ''            
    for state, score in state_score.iteritems():
        if score > highest_score:
            highest_score = score
            happiest_state = state
    print happiest_state
        
def main():
    sent_file = open(sys.argv[1])
    tweet_file = open(sys.argv[2])
    happiest_state(sent_file, tweet_file)

if __name__ == '__main__':
    main()