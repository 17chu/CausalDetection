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
# Modified to Include NP-Verb-NP detection


#Parameters
use_threshold = 0
WNVerbsOnly = 0
threshold = 7
lemmatize_candidates = 1
lemmatize_cues = 1
debug = 1
print("Use threshold: %d WNVerbsOnly: %d threshold: %d Lemmatize Candidates: %d Lemmatize Cues: %d Debug: %d" % (use_threshold, WNVerbsOnly, threshold, lemmatize_candidates, lemmatize_cues, debug))
def PipeLineTest(line):
    print("\nInput Line:\n\t%s\n" % line[:-1],end='')
    print("Converting Line to Sentence and Causal/Non-Causal Tag:\n\t" + "parseLine(line):\n\t\t",end='')
    sentence,val = parseLine(line)
    print("Sentence: %s\n\t\tNon-Causal(0)/Causal(1): %d" % (sentence,val))
    print("Analyzing sentence for Parts-Of-Speech and discourse cues:\n\t" + "Cue_Found = analyze(sentence):\n\t\t",end='')
    Cue_Found = analyze(sentence)
    print("\nSince ",end='')
    if (Cue_Found != None):
        print("Cue_Found != None \n==> discourse cue found \n==> sentence is detected as CAUSAL!")
    else:
        print("Cue_Found == None \n==> discourse cue not found in sentence \n==> sentence is detected as NON-CAUSAL!")
    print("Causal Detection was ",end='')
    if ((Cue_Found == None and val == 0) or (Cue_Found != None and val == 1)):
        print("CORRECT!")
    else:
        print("INCORRECT!")
        if (Cue_Found == None):
            print("Sentence was actually CAUSAL! (False Negative/Type 2 Error)")
        else:
            print("Sentence was actually NON-CAUSAL! (False Positive/Type 1 Error)")
    return Cue_Found
def parseLine(line):
    val = int(line[-2])
    sentence = line[:-2].replace("\"","").strip()
    return sentence,val

def matchToCue(candidate,sentence):
    if (debug == 1):
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
            print("\t\t\t\t\t" + "Attempting to match candidate verb to remaining part of discourse cue:" + "\n\t\t\t\t\t\t" + "Options:",end='')
            print(cue_dict.get(word))
        for value in cue_dict.get(word):
            if (debug == 1):
                print("\t\t\t\t\t\t\t" + "Attempting to match to: '%s'" % value)
            if (matched_string != None and matched_string != ""):
                break
            if (value == ""):
                if (debug == 1):
                    print("\t\t\t\t\t\t\t\t" + "Matched to empty string")
                    print("\t\t\t\t\t\t\t\t" + "(which means '%s' is a single-word cue)," % word)
                    print("\t\t\t\t\t\t\t\t" + "will check for other, multi-word options.")                    
                matched_string = ""
                continue
            else:
                match_to = ' '.join(sentence[candidate+1:candidate + len(value.split(' '))])
                if (debug == 1):
                    print("\t\t\t\t\t\t\t\t" + "Remaining part of sentence to match to:" + match_to)
                if (value == match_to):
                    matched_string = value
        if (matched_string == None):
            if (debug == 1):
                print("\t\t\t\t"+ "Failed to match candidate verb to discourse cue, returning None")
            return None
        else:
            if (debug == 1):
                print("\t\t\t\t"+ "Matched candidate verb/phrase to discourse cue, returning \"" + word + matched_string + "\"")
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
        print("\n\t\tUsing POS Tags to search for verbs (VBs) in sentence:")
    for i in range(0,len(POS_Tags)):
        if (len(POS_Tags[i][1]) >= 2 and POS_Tags[i][1][:2] == "VB"):
            if (debug == 1):
                print("\t\t\t",end=''),print(POS_Tags[i])
            candidates.append(i)
    if (debug == 1 and len(candidates) == 0):
        print("\t\t\t--No Candidates Found--")
    elif (debug == 1):
        print()
    if (debug == 1 and len(candidates) > 0):
        print("\t\t" + "Iterating through verb-candidates to potentially match to discourse cue:")
        print("\t\t" + "--Return Value = 'None' means verb did not match any discourse cues (therefore, verb does not imply sentence is causal)\n\t\t--Return Value != 'None' means verb matched a discourse cue and sentence is assumed causal")
    for i in candidates:
        if (debug == 1):
            print("\n\t\t\t" + "matchToCue() for candidate '" + sentence[i] + "':")
        return_val = matchToCue(i,sentence)
        if (debug == 1):
            print("\t\t\t\t" + "Return Value: ",end='')
            if (return_val == None):
                print("None")
            else:
                print(return_val)
        if (return_val != None):
            if (debug == 1):
                print("\t\t\t" + "Non-Empty Return Value (i.e. discourse cue '%s') matched, returning discourse cue" % return_val)
            return return_val
        if (debug == 1):
            print("\t\t\t" + "Discourse cue not found for current candidate")
    if (debug == 1):
        print("\t\t\t" + "Candidate verbs did not match any discourse cues, returning None'")
    return None

cue_file = "CausalCues_WithoutModifiers_WithoutComments.txt"
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
sys.exit("")
Causal_Sentences = list()
semEvalData_file = "semtest.txt"
readData = open(semEvalData_file,'r')
total = 0
correct = 0
type_1 = 0
type_2 = 0
ans = None
for line in readData:
    sentence,val = parseLine(line)
    if (debug):
        ans = PipeLineTest(line)
    else:
        ans = analyze(sentence)    
    sentence,val = parseLine(line)
    choice = 0
    if (ans != None):
        choice = 1
    if (val == choice):
        correct += 1
    else:
        if (val == 0):
            type_1 += 1
        else:
            type_2+= 1
    total += 1
    if total % 50 == 0:
        print("Sentences Evaluated: %d \t Accuracy: %f \t False Positives: %f \t False Negatives: %f" % (total,correct/total,type_1/total,type_2/total))