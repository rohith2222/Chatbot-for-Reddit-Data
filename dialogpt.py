import torch
import urllib.request
import urllib.parse
import json
import urllib

from transformers import AutoModelWithLMHead, AutoTokenizer


from classifier import Classifier
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-medium")

conversation = []


topics = {"politics": 0, "environment": 1, "technology":2, "healthcare":3,  "education": 4 , "chitchat" : 5}



def classify(input):

    c  = Classifier()

    return c.classify(input)



def get_chitchat_response(input, history=[]):
    # tokenize the new input sentence
    new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([torch.LongTensor(history), new_user_input_ids], dim=-1)

    # generate a response
    history = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id, temperature=0.6,
                             repetition_penalty=1.3).tolist()

    # convert the tokens to text, and then split the responses into lines
    response = tokenizer.decode(history[0]).split("<|endoftext|>")
    response = [(response[i], response[i + 1]) for i in range(0, len(response) - 1, 2)]

    return response, history

def get_result_from_solr(topic, input ):

    response = 'Sorry, I don\'t know about that'

    search_fields = "selftext title"

    if( topic == '' or topic == 'Generic'):
        dismaxinurl = 'http://34.132.64.242:8983/solr/IRF22P1/select?fl=id%2Cscore%2Cselftext%2Csubreddit%2Ctopic%2Ctitle&defType=dismax&indent=true&q.op=OR&q=' + urllib.parse.quote(input) + '&qf=' + urllib.parse.quote(
            search_fields) + '&rows=10'
    else:
        topic = topic.lower()
        dismaxinurl = 'http://34.132.64.242:8983/solr/IRF22P1/select?facet.field=subreddit&facet.query='+topic+'&facet=true&fl=id%2Cscore%2Cselftext%2Csubreddit%2Ctopic%2Ctitle&defType=dismax&indent=true&q.op=OR&q=' + urllib.parse.quote(
            input) + '&qf=' + urllib.parse.quote(
            search_fields) + '&rows=10'


    data = urllib.request.urlopen(dismaxinurl)
    data = json.load(data)
    docs = data['response']['docs']
    if(len(docs) != 0):
        response = docs[0]['selftext']


    return response


def chatbot(topic, input, history=[] ):


    is_chitchat = classify(input)


    if(is_chitchat ):
        response, history = get_chitchat_response(input, history)
        conversation.append(response[len(response) - 1])
    else:
        solr_res = get_result_from_solr(topic, input )
        conversation.append((input, solr_res))


    return conversation, history




