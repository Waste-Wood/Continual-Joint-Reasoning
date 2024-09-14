import os
import sys
import json
import spacy
import configparser
from tqdm import tqdm
from spacy.matcher import Matcher
import networkx as nx


blacklist = set(["from", "as", "more", "either", "in", "and", "on", "an", "when", "too", "to", "i", "do", "can", "be", "that", "or", "the", "a", "of", "for", "is", "was", 
                 "the", "-PRON-", "actually", "likely", "possibly", "want", "make", "my", "someone", "sometimes_people", "sometimes","would", "want_to", "one", "something", 
                 "sometimes", "everybody", "somebody", "could", "could_be","mine","us","em", "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "A", "a1", "a2", "a3", "a4", "ab", 
                 "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", 
                 "affecting", "after", "afterwards", "ag", "again", "against", "ah", "ain", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", 
                 "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", 
                 "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appreciate", "approximately", "ar", "are", "aren", "arent", "arise", "around", "as", 
                 "aside", "ask", "asking", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "B", "b1", "b2", "b3", "ba", "back", "bc", 
                 "bd", "be", "became", "been", "before", "beforehand", "beginnings", "behind", "below", "beside", "besides", "best", "between", "beyond", "bi", "bill", "biol", 
                 "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "C", "c1", "c2", "c3", "ca", "call", "came", 
                 "can", "cannot", "cant", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "ci", "cit", "cj", "cl", "clearly", "cm", "cn", "co", "com", "come", "comes", 
                 "con", "concerning", "consequently", "consider", "considering", "could", "couldn", "couldnt", "course", "cp", "cq", "cr", "cry", "cs", "ct", "cu", "cv", "cx", 
                 "cy", "cz", "d", "D", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "dj", "dk", 
                 "dl", "do", "does", "doesn", "doing", "don", "done", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "E", "e2", "e3", "ea", 
                 "each", "ec", "ed", "edu", "ee", "ef", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "en", "end", "ending", "enough", 
                 "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", 
                 "everywhere", "ex", "exactly", "example", "except", "ey", "f", "F", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", 
                 "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", 
                 "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "G", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", 
                 "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "H", "h2", "h3", "had", "hadn", "happens", "hardly", "has", "hasn", 
                 "hasnt", "have", "haven", "having", "he", "hed", "hello", "help", "hence", "here", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hh", "hi", "hid", 
                 "hither", "hj", "ho", "hopefully", "how", "howbeit", "however", "hr", "hs", "http", "hu", "hundred", "hy", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", 
                 "ic", "id", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "im", "immediately", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", 
                 "information", "inner", "insofar", "instead", "interest", "into", "inward", "io", "ip", "iq", "ir", "is", "isn", "it", "itd", "its", "iv", "ix", "iy", "iz", "j", "J", 
                 "jj", "jr", "js", "jt", "ju", "just", "k", "K", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "ko", "l", "L", "l2", "la", "largely", "last", "lately", "later", 
                 "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ln", "lo", 
                 "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "M", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "meantime", 
                 "meanwhile", "merely", "mg", "might", "mightn", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", 
                 "ms", "mt", "mu", "much", "mug", "must", "mustn", "my", "n", "N", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "neither", 
                 "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", 
                 "not", "noted", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "O", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", 
                 "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "otherwise", 
                 "ou", "ought", "our", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "P", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", 
                 "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", 
                 "poorly", "pp", "pq", "pr", "predominantly", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", 
                 "q", "Q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "R", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", 
                 "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", 
                 "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "S", "s2", "sa", "said", "saw", "say", "saying", "says", "sc", "sd", "se", 
                 "sec", "second", "secondly", "section", "seem", "seemed", "seeming", "seems", "seen", "sent", "seven", "several", "sf", "shall", "shan", "shed", "shes", "show", "showed", 
                 "shown", "showns", "shows", "si", "side", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somehow", "somethan", "sometime", 
                 "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", 
                 "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "sz", "t", "T", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", 
                 "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", 
                 "there", "thereafter", "thereby", "thered", "therefore", "therein", "thereof", "therere", "theres", "thereto", "thereupon", "these", "they", "theyd", "theyre", "thickv", 
                 "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", 
                 "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", 
                 "ts", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "U", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", 
                 "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "used", "useful", "usefully", "usefulness", "using", "usually", "ut", "v", "V", "va", "various", 
                 "vd", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "W", "wa", "was", "wasn", "wasnt", "way", "we", "wed", "welcome", 
                 "well", "well-b", "went", "were", "weren", "werent", "what", "whatever", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", 
                 "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "whom", "whomever", "whos", "whose", "why", "wi", 
                 "widely", "with", "within", "without", "wo", "won", "wonder", "wont", "would", "wouldn", "wouldnt", "www", "x", "X", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", 
                 "xn", "xo", "xs", "xt", "xv", "xx", "y", "Y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "your", "youre", "yours", "yr", "ys", "yt", "z", "Z", "zero", "zi", "zz"])


nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
config = configparser.ConfigParser()
config.read('')

with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
    cpnet_vocab = [l.strip() for l in list(f.readlines())]
cpnet_vocab = set([c.replace("_", " ") for c in cpnet_vocab])


# obtain entities both in sentence and conceptnet
def hard_ground(sent):
    sent = sent.lower()
    doc = nlp(sent)
    res = set()
    for t in doc:
        if t.lemma_ in cpnet_vocab and t.lemma_ not in blacklist:
            if t.pos_ == "NOUN" or t.pos_ == "VERB" or t.pos_ == "ADJ":
                res.add(t.lemma_)
    return res


# matching entities in all sentnces
def match_mentioned_concepts(sents):
    res = []
    for s in sents:
        concepts = hard_ground(s)
        res.append(concepts)
    return res


# load conceptnet index resources
def load_resources():
    global concept2id, relation2id, id2relation, id2concept, concept_embs, relation_embs
    concept2id = {}
    id2concept = {}
    with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            concept2id[w.strip()] = len(concept2id)
            id2concept[len(id2concept)] = w.strip()

    print("concept2id done")
    id2relation = {}
    relation2id = {}
    with open(config["paths"]["relation_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            id2relation[len(id2relation)] = w.strip()
            relation2id[w.strip()] = len(relation2id)

    print("relation2id done")



# load conceptnet graph
def load_cpnet():
    global cpnet,concept2id, relation2id, id2relation, id2concept, cpnet_simple
    print("loading cpnet....")
    cpnet = nx.read_gpickle(config["paths"]["conceptnet_en_graph"])
    print("Done")

    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


# get and the relation id between two entities
def get_edge(src_concept, tgt_concept):
    global cpnet, concept2id, relation2id, id2relation, id2concept
    try:
        rel_list = cpnet[src_concept][tgt_concept]
        return list(set([rel_list[item]["rel"] for item in rel_list]))
    except:
        return []



# retrieve T-hop neighbors
def find_neighbours_frequency(source_concepts, T, max_B=100):
    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple, total_concepts_id
    source = [concept2id[s_cpt] for s_cpt in source_concepts]
    start = source
    Vts = dict([(x,0) for x in start])
    Ets = {}
    total_concepts_id_set = set(total_concepts_id)
    for t in range(T):
        V = {}
        for s in start:
            if s in cpnet_simple:
                for n in cpnet_simple[s]:
                    if n not in Vts:
                        if n not in Vts:
                            if n not in V:
                                V[n] = 1
                            else:
                                V[n] += 1

                        if n not in Ets:
                            rels = get_edge(s, n)
                            if len(rels) > 0:
                                Ets[n] = {s: rels}  
                        else:
                            rels = get_edge(s, n)
                            if len(rels) > 0:
                                Ets[n].update({s: rels})  
                        
        
        V = list(V.items())
        count_V = sorted(V, key=lambda x: x[1], reverse=True)[:max_B]
        start = [x[0] for x in count_V if x[0] in total_concepts_id_set]
        
        Vts.update(dict([(x, t+1) for x in start]))
    
    _concepts = list(Vts.keys())
    _distances = list(Vts.values())
    concepts = []
    distances = []
    for c, d in zip(_concepts, _distances):
        concepts.append(c)
        distances.append(d)
    assert(len(concepts) == len(distances))
    
    tmp_triples = []
    for v, N in Ets.items():
        if v in concepts:
            for u, rels in N.items():
                if u in concepts:
                    tmp_triples.append((u, rels, v))
    
    res = [id2concept[x].replace("_", " ") for x in concepts]
    
    triples = []
    for (x, y, z) in tmp_triples:
        x = id2concept[x].replace("_", " ")
        z = id2concept[z].replace("_", " ")
        y = y[0]
        if y >= len(id2relation):
            y = y - len(id2relation)
            y = id2relation[y]
            triple = (z, y, x)
        else:
            y = id2relation[y]
            triple = (x, y, z)
        
        triples.append(triple)

          
    # triples = [(id2concept[x].replace("_", " "), [id2relation[ys] for ys in y], id2concept[z].replace("_", " ")) for (x,y,z) in triples]

    retrieved_entities = [res[i] for i in range(len(res)) if distances[i] != 0][:max_B]

    return retrieved_entities, res, distances, triples


def load_total_concepts(data_path):    
    global concept2id, total_concepts_id, config
    total_concepts = []
    total_concepts_id = []
    for path in [data_path + "/train.concepts_nv.json", data_path + "/val.concepts_nv.json"]:
        with open(path, 'r') as f:
            for line in f.readlines():
                line = json.loads(line)
                total_concepts.extend(line['qc'] + line['ac'])

        total_concepts = list(set(total_concepts))

    filtered_total_conncepts = []
    for x in total_concepts:
        if concept2id.get(x, False):
            total_concepts_id.append(concept2id[x])
            filtered_total_conncepts.append(x)
    

def read_model_vocab(data_path):
    global model_vocab
    vocab_dict = json.loads(open(data_path, 'r').readlines()[0])
    model_vocab = []
    for tok in vocab_dict.keys():
        if tok.startswith('Ġ'):
            model_vocab.append(tok[1:])

    print(len(model_vocab))


load_cpnet()
load_resources()
load_total_concepts('')




