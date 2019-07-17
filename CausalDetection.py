import nltk
import sys
from collections import defaultdict
from nltk.tag import StanfordPOSTagger
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
jar = 'stanford-postagger-2018-10-16\stanford-postagger.jar'
model = 'stanford-postagger-2018-10-16\models\english-left3words-distsim.tagger'
pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')
cue_dict = defaultdict(set)

#Parameters
use_threshold = 1
WNVerbsOnly = 0
threshold = 7
lemmatize_candidates = 1
lemmatize_cues = 1
debug = 1
#print("Use threshold: " + use_threshold  + "WNVerbsOnly: " + 
def PipeLineTest(line):
    debug = 1
    print("Input Line: %s\n" % line[:-1],end='')
    print("Converting Line to Sentence and Causal/Non-Causal Tag:\n\t" + "parseLine(line):\n\t\t",end='')
    sentence,val = parseLine(line)
    print("Sentence: %s\n\t\tNon-Causal(0)/Causal(1): %d" % (sentence,val))
    print("Analyzing sentence for Parts-Of-Speech and discourse cues:\n\t" + "analyze(sentence):\n\t\t",end='')
    analyze(sentence)
def parseLine(line):
    val = int(line[-2])
    sentence = line[:-2].replace("\"","").strip()
    return sentence,val

def matchToCue(candidate,sentence):
    print("\t\t\t\tLemmatize Verb Candidate Flag is ",end='')
    if (lemmatize_candidates == 1):
        if (debug == 1):
            print("set, lemmatizing:\n\t\t\t\t\t%s --> " % (sentence[candidate]),end='')
        word = WordNetLemmatizer().lemmatize(word=sentence[candidate],pos="v")
        if (debug == 1):
            print(word)
    else:
        word = sentence[candidate]
        if (debug == 1):
            print("not set, candidate '" + word + "' is being compared to cues directly")
    if (debug == 1):
        print("\t\t\t\t" + "Checking if word is in cue dictionary:")
    if (cue_dict.get(word) == None):
        if (debug == 1):
            print("\t\t\t\t\t" + "Candidate verb not found, returning 'None'")
        return None
    else:
        if (debug == 1):
            print("\t\t\t\t" + "Matched candidate verb to causal cue list")
        matched_string = None
        if (debug == 1):
            print("\t\t\t\t" + "Attempting to match candidate verb to remaining part of discourse cue:\n\t\t\t\t\tOptions:",end='')
            print(cue_dict.get(word))
        for value in cue_dict.get(word):
            if (matched_string != None and matched_string != ""):
                break
            if (value == ""):
                matched_string = ""
                continue
            else:
                match_to = ' '.join(sentence[candidate+1:candidate + len(value.split(' '))])
                #print("Matching \"" + value + "\" to sentence part \"" + match_to + "\"")
                if (value == match_to):
                    matched_string = value
        if (matched_string == None):
            return None
        else:
            #print("Matched candidate value to: \"" + word + "\"")
            if (matched_string != ""):
                return word + " " + matched_string
            else:
                return word
                
def analyze(sentence):
    if (debug == 1):
        print("Tagging using Stanford POS Tagger ('pos_tagger.tag(word_tokenize(sentence))')\n\t\t\t",end = '')
    POS_Tags = pos_tagger.tag(word_tokenize(sentence))
    if (debug == 1):
        print(POS_Tags)
    sentence = list()
    for i in range(0,len(POS_Tags)):
        sentence.append(POS_Tags[i][0])
    if (debug == 1):
        print("\t\tSentence converted to word-tokenized list (mimicking POS-Tagger tokenization):\n\t\t\t",end=''),print(sentence)
    #print("Sentence:")
    #print(sentence)
    candidates = list()
    #print("Candidates:")
    if (debug == 1):
        print("\t\tUsing POS Tags to search for verbs (VBs) in sentence:")
    for i in range(0,len(POS_Tags)):
        if (len(POS_Tags[i][1]) >= 2 and POS_Tags[i][1][:2] == "VB"):
            if (debug == 1):
                print("\t\t\t",end=''),print(POS_Tags[i])
            candidates.append(i)
    if (debug == 1 and len(candidates) == 0):
        print("\t\t\t--No Candidates Found--")
    if (debug == 1 and len(candidates) > 0):
        print("\t\t" + "Iterating through verb-candidates to potentially match to discourse cue:")
        print("\t\t" + "  Return Value = 'None' means verb did not match any discourse cues (therefore, verb does not imply sentence is causal)\n\t\t  Return Value != 'None' means verb matched a discourse cue and sentence is assumed causal")
    for i in candidates:
        if (debug == 1):
            print("\t\t\t" + "matchToCue() for candidate '" + sentence[i] + "':")
        return_val = matchToCue(i,sentence)
        if (debug == 1):
            print("\t\t\t\t" + "Return Value: ",end='')
            if (return_val == None):
                print("None ")
            else:
                print(return_val)
        if (return_val != None):
            return return_val
    return None

cue_file = "girju.txt"
inFile = open(cue_file)
for line in inFile:
    line = line.replace('\n',"").strip()
    line = line.split(' ')
    if (lemmatize_cues == 1):
        verb = WordNetLemmatizer().lemmatize(line[0],'v')
    else:
        verb = line[0]
    if (use_threshold == 1):
        count = 0
        synsets = wn.synsets(verb)
        if (WNVerbsOnly == 1):
            synsets = wn.synsets(verb,pos=wn.VERB)
        for synset in synsets:
            if verb in synset.name():
                count += 1
        if (count > use_threshold or use_threshold == 0):
            continue            
    if (len(line) == 1):        
        cue_dict[verb].add("")
    else:
        cue_dict[verb].add(' '.join(line[1:]))
        
Causal_Sentences = list()
semEvalData_file = "semtest.txt"
readData = open(semEvalData_file,'r')
total = 0
correct = 0
type_1 = 0
type_2 = 0
ans = None
sys.exit("")
for line in readData:
    sentence,val = parseLine(line)
    ans = analyze(sentence)
    choice = 0
    if (ans != None):
        choice = 1
    if (val == choice):
        #print("CORRECT:")
        #if (val == 0):
            #print("\tNON-CAUSAL")
        #else:
            #print("\tCAUSAL")
        #print('\t' + "\"" + sentence + "\"")
        correct += 1
    else:
        #print("INCORRECT:")
        if (val == 0):
            #print("\tSupposed to be NON-CAUSAL(Type 1 Error/False Positive)")
            type_1 += 1
        else:
            #print("\tSupposed to be CAUSAL (Type 2 Error/False Negative)")
            type_2+= 1
        #print('\t' + "\"" + sentence + "\"")
    #if (ans != None):
        #print("\tCue: \"" + ans + "\"")
    total += 1
    if total % 100 == 0:
        print("Total is currently: " + str(total))
print("\nSentences Evaluated: %d \t Accuracy: %f \t False Positives: %f \t False Negatives: %f" % (total,correct/total,type_1/total,type_2/total))