from numpy import argmax
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BartForConditionalGeneration, BartTokenizer
from module.kuma import KumaAttention
import torch.nn.functional as F
from torch.nn.utils import rnn
from utils.tools import filter_pos, retrieve
from nltk.corpus import stopwords
import json
from nltk.tokenize import word_tokenize
from nltk import pos_tag_sents
from collections import OrderedDict
import copy
import pdb


blacklist = ["from", "as", "more", "either", "in", "and", "on", "an", "when", "too", "to", "i", "do", "can", "be", "that", "or", "the", "a", "of", "for", "is", "was", 
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
            "xn", "xo", "xs", "xt", "xv", "xx", "y", "Y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "your", "youre", "yours", "yr", "ys", "yt", "z", "Z", "zero", "zi", "zz"]


def get_z_counts(att, prem_mask, hypo_mask):
    """
    Compute z counts (number of 0, continious, 1 elements).
    :param att: similarity matrix [B, prem, hypo]
    :param prem_mask:
    :param hypo_mask:
    :return: z0, zc, z1
    """
    # mask out all invalid positions with -1
    att = torch.where(hypo_mask.unsqueeze(1), att, att.new_full([1], -1.))
    att = torch.where(prem_mask.unsqueeze(2), att, att.new_full([1], -1.))

    z0 = (att == 0.).sum().item()
    zc = ((0 < att) & (att < 1)).sum().item()
    z1 = (att == 1.).sum().item()

    assert (att > -1).sum().item() == z0 + zc + z1, "mismatch"

    return z0, zc, z1


def get_embedding(hps, model, input_ids):
    if hps.model_name == 'bart':
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        token_embed = model.get_encoder().embed_tokens(input_ids) * model.get_encoder().embed_scale
        return token_embed
    elif hps.model_name == 't5':
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        token_embed = model.encoder.embed_tokens(input_ids)
        return token_embed
    else:
        return None


def process_text(sent: str):
    if '.' in sent and not sent.startswith('.'):
        sent = sent.split('.')[0] + '.'
    while '\n\n' in sent:
        sent = sent.replace('\n\n', '\n')
    sent = sent.replace('\n', ' ')
    return sent.strip()


def masked_softmax(t, mask, dim=-1):
    t = torch.where(mask, t, t.new_full([1], float('-inf')))
    return F.softmax(t, dim=dim)



class MultiHeadAttenInduction(nn.Module):
    def __init__(self, hps, hidden_dim, tokenizer):
        super(MultiHeadAttenInduction, self).__init__()
        self.hps = hps
        self.tokenizer = tokenizer
        self.dim = hidden_dim // hps.num_heads
        self.q = nn.Linear(self.dim, self.dim, bias=False)
        self.k = nn.Linear(self.dim, self.dim, bias=False)
        self.v = nn.Linear(self.dim, self.dim, bias=False)
        self.sqrt_d = pow(self.dim, 0.5)
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.gelu = nn.GELU()
    
    def forward(self, batch_word, batch_knowledge, model, mode, embeds=False, atten_mask=None):
        cause, effect = batch_word
        if mode == 'abductive':
            inputs, outputs = effect, cause
        else:
            inputs, outputs = cause, effect
        
        if not embeds:
            inputs_t = self.tokenizer(inputs, padding=True, return_tensors='pt')
            inputs_t = [inputs_t[term].cuda(self.hps.gpu[0]) for term in inputs_t]
            input_embed = get_embedding(self.hps, model, inputs_t[0])
        else:
            inputs_t = [inputs, atten_mask]
            input_embed = inputs

        outputs_t = self.tokenizer(outputs, padding=True, return_tensors='pt')
        outputs_t = [outputs_t[term].cuda(self.hps.gpu[0]) for term in outputs_t]

        kg_input_ids, kg_atten_mask = self.tokenizer(batch_knowledge, padding=True, return_tensors='pt').values()

        kg_input_ids = kg_input_ids.cuda(self.hps.gpu[0])
        kg_atten_mask = kg_atten_mask.cuda(self.hps.gpu[0])
        kg_embed = get_embedding(self.hps, model, kg_input_ids)

        input_atten_mask = (1.0 - inputs_t[1]) * -100000000
        kg_atten_mask = (1.0 - kg_atten_mask) * -100000000

        atten_heads = []
        for i in range(self.hps.num_heads):
            q = self.q(input_embed[:, :, i*self.dim:(i+1)*self.dim])
            k = self.k(kg_embed[:, :, i*self.dim:(i+1)*self.dim]).transpose(1, 2)
            v = self.v(kg_embed[:, :, i*self.dim:(i+1)*self.dim])

            atten_score = q @ k / self.sqrt_d
            atten_score = atten_score.permute(2, 0, 1) + input_atten_mask
            atten_score = (atten_score.permute(2, 1, 0) + kg_atten_mask).permute(1, 0, 2)

            atten_heads.append(torch.softmax(atten_score, -1) @ v)
        
        atten_res = torch.cat(atten_heads, -1)
        atten_add_norm = self.layer_norm(self.gelu(self.linear(atten_res)) + input_embed)
        
        labels = torch.where(outputs_t[0] == self.tokenizer.pad_token_id, -100, outputs_t[0])

        return atten_add_norm, inputs_t[1], labels, outputs_t[1]


class HardKumaInduction(nn.Module):
    def __init__(self, hps, hidden_dim, tokenizer):
        super(HardKumaInduction, self).__init__()
        self.hps = hps
        self.attention = KumaAttention(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.gelu = nn.GELU()
        self.tokenizer = tokenizer
        self.prem_mask = None
        self.hypo_mask = None
        self.prem2hypo_att = None
        self.hypo2prem_att = None
        self.lambda_init = 1e-5
        self.register_buffer('lambda0', torch.full((1,), self.lambda_init))
        self.register_buffer('c0_ma', torch.full((1,), 0.)) 
        self.selection = 0.10
        self.lagrange_lr = 0.01
        self.lagrange_alpha = 0.99

    def _mask_padding(self, x, mask, value=0.):
        """
        Mask should be true/1 for valid positions, false/0 for invalid ones.
        :param x:
        :param mask:
        :return:
        """
        return torch.where(mask, x, x.new_full([1], value))

    def forward(self, batch_word, batch_knowlwdge, model, mode, embeds=False, atten_mask=None):
        cause, effect = batch_word
        if mode == 'abductive':
            inputs, outputs = effect, cause
        else:
            inputs, outputs = cause, effect
        
        if not embeds:
            inputs_t = self.tokenizer(inputs, padding=True, return_tensors='pt')
            inputs_t = [inputs_t[term].cuda(self.hps.gpu[0]) for term in inputs_t]
            input_embed = get_embedding(self.hps, model, inputs_t[0])
        else:
            inputs_t = [inputs, atten_mask]
            input_embed = inputs

        
        outputs_t = list(self.tokenizer(outputs, padding=True, return_tensors='pt').values())
        kg_input_ids, kg_atten_mask = self.tokenizer(batch_knowlwdge, padding=True, return_tensors='pt').values()

        kg_input_ids = kg_input_ids.cuda(self.hps.gpu[0])
        kg_atten_mask = kg_atten_mask.cuda(self.hps.gpu[0])
        
        outputs_t = [term.cuda(self.hps.gpu[0]) for term in outputs_t]

        kg_embed = get_embedding(self.hps, model, kg_input_ids)

        self.input_mask = input_mask = (inputs_t[1] != 0)
        self.kg_mask = kg_atten_mask = (kg_atten_mask != 0)
        self.prem_mask = input_mask
        self.hypo_mask = kg_atten_mask

        atten_score = self.attention(input_embed, kg_embed)

        self.prem2hypo_att = atten_score
        self.hypo2prem_att = atten_score.transpose(1, 2)

        atten_score = self._mask_padding(atten_score, kg_atten_mask.unsqueeze(1), 0.)
        atten_score = self._mask_padding(atten_score, input_mask.unsqueeze(2), 0.)

        self.word2kg_att = masked_softmax(atten_score, kg_atten_mask.unsqueeze(1))
        attented_word = self.word2kg_att @ kg_embed

        attented_word_add_norm = self.layer_norm(self.gelu(self.linear(attented_word)) + input_embed)

        labels = torch.where(outputs_t[0] == self.tokenizer.pad_token_id, -100, outputs_t[0])

        return attented_word_add_norm, inputs_t[1], labels, outputs_t[1]
    
    def get_loss(self):
        optional = OrderedDict()

        # training stats
        if self.training:
            # note that the attention matrix is symmetric now, so we only
            # need to compute the counts for prem2hypo
            z0, zc, z1 = get_z_counts(
                self.prem2hypo_att, self.prem_mask, self.hypo_mask)
            zt = float(z0 + zc + z1)
            optional["p2h_0"] = z0 / zt
            optional["p2h_c"] = zc / zt
            optional["p2h_1"] = z1 / zt
            optional["p2h_selected"] = 1 - optional["p2h_0"]

        # regularize sparsity
        assert isinstance(self.attention, KumaAttention), \
            "expected HK attention for this model, please set dist=hardkuma"

        if self.selection > 0:

            # Kuma attention distribution (computed in forward call)
            z_dist = self.attention.dist

            # pre-compute pdf(0)  shape: [B, |prem|, |hypo|]
            pdf0 = z_dist.pdf(0.)
            pdf0 = pdf0.squeeze(-1)

            prem_lengths = self.prem_mask.sum(1).float()
            hypo_lengths = self.hypo_mask.sum(1).float()


            pdf_nonzero = 1. - pdf0  # [B, T]
            pdf_nonzero = self._mask_padding(
                pdf_nonzero, mask=self.hypo_mask.unsqueeze(1), value=0.)
            pdf_nonzero = self._mask_padding(
                pdf_nonzero, mask=self.prem_mask.unsqueeze(2), value=0.)

            l0 = pdf_nonzero.sum(2) / (hypo_lengths.unsqueeze(1) + 1e-9)
            l0 = l0.sum(1) / (prem_lengths + 1e-9)
            l0 = l0.sum() / self.prem_mask.size(0)

            c0_hat = (l0 - self.selection)

            # moving average of the constraint
            self.c0_ma = self.lagrange_alpha * self.c0_ma + (1 - self.lagrange_alpha) * c0_hat.item()

            # compute smoothed constraint (equals moving average c0_ma)
            c0 = c0_hat + (self.c0_ma.detach() - c0_hat.detach())

            # update lambda
            self.lambda0 = self.lambda0 * torch.exp(self.lagrange_lr * c0.detach())

            with torch.no_grad():
                optional["cost0_l0"] = l0.item()
                optional["target0"] = self.selection
                optional["c0_hat"] = c0_hat.item()
                optional["c0"] = c0.item()  # same as moving average
                optional["lambda0"] = self.lambda0.item()
                optional["lagrangian0"] = (self.lambda0 * c0_hat).item()
                optional["a"] = z_dist.a.mean().item()
                optional["b"] = z_dist.b.mean().item()

            loss = self.lambda0.detach() * c0

        return loss
    

class ChainInduction(nn.Module):
    def __init__(self, hps, hidden_dim, tokenizer):
        super(ChainInduction, self).__init__()
        self.hps = hps
        self.tokenizer = tokenizer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim//2, num_layers=1, batch_first=True, dropout=0.1, bidirectional=True) 
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.gelu = nn.GELU()   

    def forward(self, batch_word, batch_knowledge, model, mode, embeds=False, atten_mask=None):
        cause, effect = batch_word
        batch_knowledge = [k.split(';') for k in batch_knowledge]
        kg_len = [len(k) for k in batch_knowledge]
        kg_nums = [sum(kg_len[:i]) for i in range(1, len(kg_len)+1)]
        kg_nums = [0] + kg_nums
        kg = []
        for k in batch_knowledge:
            kg += k
    
        if mode == 'abductive':
            inputs, outputs = effect, cause
        else:
            inputs, outputs = cause, effect
        
        if not embeds:
            inputs_t = self.tokenizer(inputs, padding=True, return_tensors='pt')
            inputs_t = [inputs_t[term].cuda(self.hps.gpu[0]) for term in inputs_t]
            input_embed = get_embedding(self.hps, model, inputs_t[0])
        else:
            inputs_t = [inputs, atten_mask]
            input_embed = inputs

        outputs_t = self.tokenizer(outputs, padding=True, return_tensors='pt')
        kg_input_ids = self.tokenizer(kg, padding=False).input_ids

        outputs_t = [outputs_t[term].cuda(self.hps.gpu[0]) for term in outputs_t]
        
        kg_embed = [get_embedding(self.hps, model, torch.LongTensor(input_ids).unsqueeze(0).cuda(self.hps.gpu[0])).squeeze(0) for input_ids in kg_input_ids]

        padded_sequences = rnn.pad_sequence(kg_embed)
        lstm_input = rnn.pack_padded_sequence(padded_sequences, [k.size(0) for k in kg_embed], enforce_sorted=False)

        lstm_output, _ = self.lstm(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output)

        dim = model.config.d_model//2
        chain_embed = [torch.cat([embed[0, dim:], embed[l-1, :dim]], -1) for embed, l in zip(unpacked.transpose(0, 1), unpacked_len)]

        chain_by_example = [chain_embed[kg_nums[i-1]: kg_nums[i]] for i in range(1, len(kg_nums))]

        lengths = [len(chains) for chains in chain_by_example]
        max_len = max(lengths)

        chains_mask = torch.LongTensor([[1]*len(chains)+[0]*(max_len-len(chains)) for chains in chain_by_example]).to(input_embed.device)
        chains_mask = (1.0 - chains_mask) * -10000000.0
        padded_chains = [chains + [torch.LongTensor([0]*2*dim).to(chains[0].device)]*(max_len-len(chains)) for chains in chain_by_example]

        padded_chains = [torch.stack(chains, 0) for chains in padded_chains]
        padded_chains = torch.stack(padded_chains, 0)

        atten_score = input_embed @ padded_chains.transpose(1, 2)
        atten_score = (atten_score.permute(1, 0, 2) + chains_mask).permute(1, 0, 2)
        atten_weight = torch.softmax(atten_score, -1)
        atten = atten_weight @ padded_chains

        linear_add = self.gelu(self.linear(atten)) + input_embed
        linear_add_norm = self.layer_norm(linear_add)

        labels = torch.where(outputs_t[0] == self.tokenizer.pad_token_id, -100, outputs_t[0])

        return linear_add_norm, inputs_t[1], labels, outputs_t[1]
    


all_relations = [
    "AtLocation",
    "CapableOf",
    "Causes",
    "CausesDesire",
    "CreatedBy",
    "DefinedAs",
    "DesireOf",
    "Desires",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPainCharacter",
    "HasPainIntensity",
    "HasPrerequisite",
    "HasProperty",
    "HasSubEvent",
    "HasSubevent",
    "HinderedBy",
    "InheritsFrom",
    "InstanceOf",
    "IsA",
    "LocatedNear",
    "LocationOfAction",
    "MadeOf",
    "MadeUpOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "NotHasA",
    "NotHasProperty",
    "NotIsA",
    "NotMadeOf",
    "ObjectUse",
    "PartOf",
    "ReceivesAction",
    "RelatedTo",
    "SymbolOf",
    "UsedFor",
    "isAfter",
    "isBefore",
    "isFilledBy",
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xReason",
    "xWant",
    ]


class RuleInduction():
    def __init__(self, hps):
        self.hps = hps
        self.stpwords = list(set(stopwords('english') + blacklist))
        self.stopwords += ['man', 'person', 'male', 'female', 
               'human', 'female human', 'male human', 
               'female person', 'male person', 'men',
               'woman', 'women', 'kid', 'child', 'children',
               'boy', 'girl', 'kids', 'boys', 'girls']
        
        indexes = json.load(open('', 'r'))
        self.entity2id, self.id2entity, self.entity2children, self.entity2parents = [indexes[key] for key in indexes]

    def forward(self, sentences):
        sentences = [word_tokenize(s) for s in sentences]
        sentences = [[word for word in words if word not in self.stpwords] for words in sentences]
        pos_tags = pos_tag_sents(sentences)

        sentences = filter_pos(pos_tags)

        entities = [retrieve(sent, self.entity2id, self.id2entity, self.entity2children, self.entity2parents, 3) for sent in sentences]
        return entities, None



