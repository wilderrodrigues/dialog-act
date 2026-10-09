

from pathlib import Path
from uu.msc.ai.mair.dialog.core.datasets import DatasetFactory

# inlezen data

# 1. Loading the data through the DatasetFactory class
dataset_path = Path("data/dialog_acts.dat")
df = DatasetFactory.load_dataframe(dataset_path, separator=" ")

# 2. Splitting in utterances and dialog acts
utterances = df["utterance"].tolist()
dialog_acts = df["act"].tolist()

# The keywords

keywords = {
    "thankyou" : ["thank", "thank you bye"],
"affirm" : ["yes"],
"negate"   : ["no", "no im looking for", "no i am looking for", "no i want", "no i need", "no i would", "no in", "no id like"],
"inform"  :  ["dont", "any", "im looking for", "i want", "i need", "no id like", "can i find", "im looking"],
"reqalts"  : ["is there", "are there", "how about", "else", "about"],
"request"  : ["what is","phone number", "address", "post code"],
"restart"  : ["start again", "start", "reset"],
"reqmore" : ["more"],
"repeat"  : ["repeat", "back"],
"ack"  : ["fine", "kay", "okay", "well"],
"bye"  : ["good bye", "goodbye"],
"deny" : [ "wrong",],
"hello"  : ["hi", "hello", "hey"],
"confirm" : ["does it", "is it", "do they", "is this"],
"null" : [],
}



# set maken met alle trefwoorden, los gevolgd door een dialog_act
# daarvoor moeten we twee for loops hebben, omdat we anders een set
# namelijk meerdere woorden
# plaatsen in de set, en daar kunnen we niet zoveel mee. 
# Daarom plaatsen we een tuple. 

all_lines = []

for dialog_act, trefwoorden in keywords.items():
    for trefwoord in trefwoorden:
       all_lines.append((trefwoord, dialog_act)) 

#all_lines.sort(key=lambda x: len(x[0]), reverse=True)
#sorteren de woorden van groot naar klein, omdat je bijv 'no' 
# later wilt onderzoeken dan no i dont.
# THis line (51) is commented out, because we first did use it, than we didn't
# and the accuracy dramatically increased

#checken of het werkt met één zin
def predict_utterance(utterances, all_lines):
    for trefwoord, act in all_lines:
        if trefwoord in utterances:
            return act

    return "inform"

# nu door alle regels gaan

correct = 0
for i in range(len(utterances)):
    exact_zin = utterances[i]
    exacte_dialog = dialog_acts[i]

    predicted_act = predict_utterance(exact_zin, all_lines)

    if predicted_act == exacte_dialog:
        correct += 1

accuracy = correct / len(utterances)
print(f"Aantal goed: {correct} van de {len(utterances)}")
print(f"Accuracy: {accuracy * 100:.2f}%")
        










# groeperen data 85 % / 15 %

