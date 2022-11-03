from re import search
from nltk.stem.lancaster import LancasterStemmer
import pickle
import json
import random
import tensorflow
import tflearn
import numpy
import nltk
import requests
from bs4 import BeautifulSoup


nltk.download('punkt')
stemmer = LancasterStemmer()

# Load database
with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph()

#
# Run training data through neural networks and save model
#
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.01)


# TensorBoard is used to visualise the accuracy and error rate of the model
# tensorboard_dir can be set to a different directory if needed. Just make sure to change it in both lines below
model = tflearn.DNN(net, tensorboard_verbose=3, tensorboard_dir='/tmp/tflearn_logs/')
model.save('/tmp/tflearn_logs/')


# If model needs to be trained, un-comment below block of code to train the model
# Can save the model with a name to load on TensorBoard here
# model.fit(training, output, n_epoch=80, batch_size=8,
#           show_metric=True, run_id='NAME_OF_TEST_HERE')
# model.save("model.tflearn")

# If model is already trained, comment above block of code and un-comment the line below
model.load("model.tflearn")

# To see the model in TensorBoard, run this code in the python terminal. Note: If tensorboard_dir was changed it will also need to be changed here
# tensorboard --logdir=/tmp/tflearn_logs/ --host localhost

# To test using TensorBoard we found it best to comment the 'chat()' function and train the model each time, setting the run_id to something memorable. 'epoch50_learningRate0.01' for example.


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

#
# Chat function: this is where the main chatbot content takes place
# Add additional function calls here like in our examples of "threats" and "industry"
#
def chat():
    print("Start talking with the chatbot now! (Type quit to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        while numpy.sum(bag_of_words(inp, words)) == 0:
            print("I can't recoginze your words, please retry:")
            inp = input("You: ")
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        print("Tag = " + tag)

        # This is where additional functions can be called
        # Add conditions to match a tag in the intent file to call functions in this way 
        if (tag == "threats"):
            threats()
        elif (tag == "industry"):
            industy()

        else:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]
            print("Ashwin: " + random.choice(responses))

#
# Function used for checking if a pattern matches an object in the database
#
def find_answer(tag):
    for tg in data["intents"]:
        if tg["tag"] == tag:
            responses = tg["responses"]
            return responses

#
# Function used to return cost of tag provided
#
def find_cost(tag):
    for tg in data["intents"]:
        if tg["tag"] == tag:
            print("in cost tag check")
            cost = tg["cost"]
            return cost

#
# Function used for determining if threat was high priority. If so, value is added to 2nd array
#
def check_cost_priority(tag):
    for tg in data["intents"]:
        if tg["tag"] == tag:
            priority = tg["priority"]
            cost = tg["cost"]
            if priority == "High":
                return cost
            else:
                return 0

#
# This function is used to show the top threats for a selected industry
# It can be altered to gather data from other sources such as if you had an API you could hit
#
def industy():
    # array to store user's input
    cyberThreats = ['Malware', 'Denial of Service',
                    'Ransomware', 'Spam', 'Phishing']
    print("Ashwin: To help answer this, I'll need know what industry you are in.")
    answered = False
    while (answered != True):
        inp = input("You: ")
        if inp.lower() != "":
            print(f"Ashwin: For the {inp} industry, the threats are a mix of {random.choice(cyberThreats)}, {random.choice(cyberThreats)} and {random.choice(cyberThreats)}. However here are some search results that you can look at.")
            print("\n")
            query = (f"Cyber threats in the {inp} industry")
            search(query)
            answered = True
            return
        else:
            print("Ashwin: You haven't told me an industry. Please try again.")


#
# This function is used in our industry function to perform our search for top threats in the industry
#
def search(query):

    search = query.replace(' ', '+')
    results = 10
    url = (f"https://www.google.com/search?q={search}&num={results}")

    requests_results = requests.get(url)
    soup_link = BeautifulSoup(requests_results.content, "html.parser")
    links = soup_link.find_all("a")

    for link in links:
        link_href = link.get('href')
        if "url?q=" in link_href and not "webcache" in link_href:
            title = link.find_all('h3')
            if len(title) > 0:
                print(title[0].getText())
                print(link.get('href').split("?q=")[1].split("&sa=U")[0])
                print("\n")

#
# This function follows the process of determining a users threats and returns a mitigation for each one
#
def threats():
    # array to store user's input
    traResults = []
    # array to store mitigation responses
    mitigations = []
    # array to store cost values
    cost = []
    high_threats_cost = []
    print("Ashwin: To help answer this, I'll need some additional information from your most recent Threat and Risk Assessment(TRA).")
    answered = False
    while (answered != True):
        print("Ashwin: Do you have a current TRA? (Yes/No)")
        inp = input("You: ")
        if inp.lower() == "no":
            print("Ashwin: You have not yet performed a TRA and therefore I am unable to answer your question.  Please perform a Threat and Risk Assessment as soon as possible in order to identify the key cyber security threats faced by your organisation.")
            print("Ashwin: Please contact us here https://securemation.com/contact-us/ to find more information for this.")
            answered = True
            return
        print("Ashwin: Was the TRA completed in the last 6 months? (Yes/No)")
        inp = input("You: ")
        if inp.lower() == "no":
            print("Ashwin: Since the TRA was performed more then 6 months ago, I would strongly recommend that you undertake another TRA ASAP.")
            print("Ashwin: However, we can still continue.")
        coreInformation = False
        print(
            "Ashwin: Could you please tell me the TRA details of your core information systems, one threat at a time.")
        while (coreInformation != True):
            inp = input("You: ")
            if inp.lower() == "no":
                coreInformation = True
            elif numpy.sum(bag_of_words(inp, words)) == 0:
                traResults.append(inp)
                mitigations.append("N/A")
                print(
                    "Ashwin: Is there anything else you wanted to add? If so, please continue or just say 'no' if you're ready to move on.")
            elif numpy.sum(bag_of_words(inp, words)) != 0:
                traResults.append(inp)
                results = model.predict([bag_of_words(inp, words)])
                results_index = numpy.argmax(results)
                tag = labels[results_index]
                mitigations.append(find_answer(tag))
                cost.append(find_cost(tag))
                high_threats_cost.append(check_cost_priority(tag))

                print(
                    "Ashwin: Is there anything else you wanted to add? If so, please continue or just say 'no' if you're ready to move on.")
        supportingPeople = False
        print(
            "Ashwin: Thankyou for that. Now please tell me what are the TRA details of your supporting people?")
        while (supportingPeople != True):
            inp = input("You: ")
            if inp.lower() == "no":
                supportingPeople = True
            elif numpy.sum(bag_of_words(inp, words)) == 0:
                traResults.append(inp)
                mitigations.append("N/A")
                print(
                    "Ashwin: Is there anything else you wanted to add? If so, please continue or just say 'no' if you're ready to move on.")
            elif numpy.sum(bag_of_words(inp, words)) != 0:
                traResults.append(inp)
                results = model.predict([bag_of_words(inp, words)])
                results_index = numpy.argmax(results)
                tag = labels[results_index]
                mitigations.append(find_answer(tag))
                cost.append(find_cost(tag))
                high_threats_cost.append(check_cost_priority(tag))

                print(
                    "Ashwin: Is there anything else you wanted to add? If so, please continue or just say 'no' if you're ready to move on.")
        supportingProcesses = False
        print(
            "Ashwin: Thankyou for that. Now please tell me what are the TRA details of your supporting processes?")
        while (supportingProcesses != True):
            inp = input("You: ")
            if inp.lower() == "no":
                supportingProcesses = True
            elif numpy.sum(bag_of_words(inp, words)) == 0:
                traResults.append(inp)
                mitigations.append("N/A")
                print(
                    "Ashwin: Is there anything else you wanted to add? If so, please continue or just say 'no' if you're ready to move on.")
            elif numpy.sum(bag_of_words(inp, words)) != 0:
                traResults.append(inp)
                results = model.predict([bag_of_words(inp, words)])
                results_index = numpy.argmax(results)
                tag = labels[results_index]
                mitigations.append(find_answer(tag))
                cost.append(find_cost(tag))
                high_threats_cost.append(check_cost_priority(tag))

                print(
                    "Ashwin: Is there anything else you wanted to add? If so, please continue or just say 'no' if you're ready to move on.")
        answered = True
    print('Ashwin: Thank you for providing the information. So from what I can see, mitigations that I would recommend for your threats are as follows:')
    for i in range(len(mitigations)):
        if mitigations[i] != "N/A":
            print(
                f'Ashwin: To resolve threat #{(i + 1)}. "{traResults[i]}" I recommend: {str(mitigations[i])}')
        else:
            print(
                f'Ashwin: No mitigation found for {traResults[i]}. Please contact us here https://securemation.com/contact-us/ for further assistance.')

    answered = False
    while (answered != True):
        print(
            "Ashwin: Would you like to explore the cost of fixing these threats? (Yes/No)")
        inp = input("You: ")
        if inp.lower() == "no":
            print("Ashwin: No worries, please contact us here https://securemation.com/contact-us/ if you change your mind.")
            answered = True
        elif inp.lower() == "yes":
            costs(cost, high_threats_cost)
            answered = True

#
# This function is given 2 arrays: 1 is the total cost to mitigate all threats, the other is cost to mitigate high prioriy threats
# Additional options can be added here to reduce the cost, such as if a user has a type of security in place, -$XX.XX from cost
#
def costs(cost, high_threats_cost):
    cost_total = 0
    high_priority_total = 0
    for i in range(0, len(cost)):
        cost_total = cost_total + cost[i]
    for i in range(0, len(high_threats_cost)):
        high_priority_total = high_priority_total + high_threats_cost[i]
    print(
        f'Ashwin: The current cost to mitigate all of your threats comes to ${cost_total}')
    print(
        f'Ashwin: If you were just wanting to mitigate your high priority threats, it would cost ${high_priority_total}')
    print(f'Ashwin: For a more specific figure tailored to your company, please contact us here https://securemation.com/contact-us/ for further assistance.')


#
# This function call is used to start the chatbot program. If you would like to plot and show TensorBoard graphs, comment the 'chat()' function to prevent chatbot opening
#
chat()
