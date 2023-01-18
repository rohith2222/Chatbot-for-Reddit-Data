import csv
import json
import re
import string

import chitchat_dataset as ccc

dataset = ccc.Dataset()

topics = {"politics": 0, "environment": 1, "technology":2, "healthcare":3,  "education": 4 , "chitchat" : 5}
alphanumspace_list = list(string.ascii_letters + string.digits + ' ')

# Opening JSON file
f = open('project1_index_details.json')

# returns JSON object as
# a dictionary
data = json.load(f)

wf = open('processed_data.csv', 'w')

writer = csv.writer(wf)


def remove_junk_chars( text):
    new_text = ''
    for char in text:
        if char in alphanumspace_list:
            new_text = new_text + char
        else:
            new_text = new_text + " "

    return new_text


# Iterating through the json
# list


writer.writerow(["sno", "text", "label"])

counter = 0
topic_count = {"politics": 0.0, "environment": 0.0, "technology":0.0, "healthcare":0.0,  "education": 0.0 , "chitchat" : 0.0}

for i in data:
    if(i["is_submission"] and  i["subreddit"] in topics ):

            text = remove_junk_chars(i["selftext"] +" " + i["title"])
            row = [counter, text , topics[i["subreddit"]]]
            writer.writerow(row)
            topic_count[i["subreddit"]] +=  topic_count[i["subreddit"]] + 1
            counter += 1

        # else:
        #     text = remove_junk_chars(i["body"] + " " + i["parent_body"])
        #     row = [counter, text, topics[i["subreddit"]]]
        #     writer.writerow(row)
        #     topic_count[i["subreddit"]] +=  topic_count[i["subreddit"]] +1








for convo_id, convo in dataset.items():


    text = ''
    for c in convo['messages']:
        for d in c:
            text = text + ' ' + d['text']
    row = [counter, text, topics["chitchat"]]
    writer.writerow(row)
    counter += 1
    topic_count["chitchat"] += topic_count["chitchat"] + 1

print(topic_count)
# Closing file
f.close()
wf.close()