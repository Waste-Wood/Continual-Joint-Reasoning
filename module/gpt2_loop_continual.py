from cmath import log
import json
import torch.nn as nn
from transformers import GPT2LMHeadModel
from nltk import word_tokenize
from nltk import pos_tag_sents
from nltk.corpus import stopwords
from utils.tools import tokenize_gpt2, tokenize4gen
import pdb
import torch
from utils.rule_retrieval import find_neighbours_frequency, match_mentioned_concepts
from .generate import sample_sequence_continual2, sample_sequence_continual2_2, sample_sequence_continual2_1, sample_sequence
from torch.autograd import Variable
from fuzzywuzzy import fuzz


loss_ce = nn.CrossEntropyLoss(reduction='mean')
loss_nll = nn.NLLLoss(reduction='mean')
softmax = nn.LogSoftmax(-1)


def process_text(sent: str):
    if '.' in sent and not sent.startswith('.'):
        sent = sent.split('.')[0] + '.'
    while '\n\n' in sent:
        sent = sent.replace('\n\n', '\n')
    sent = sent.replace('\n', ' ')
    return sent.strip()


def get_embedding(hps, model, input_ids):
    if hps.model_name == 'gpt2':
        # position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
        token_embed = model.state_dict()['transformer.wte.weight'][input_ids]
        # pos_embed = model.state_dict()['transformer.wpe.weight'][position_ids]
        # return torch.dropout(token_embed + pos_embed, 0.1, train=True)
        return token_embed
    else:
        return None


def get_mask(cause, effect, mode):
    if mode == 'deductive':
        single, multi = cause, effect
    else:
        multi, single = cause, effect

    results = []
    for i, mids in enumerate(multi):
        src = single[i]
        tmp_mask = []    
        for mid in mids:
            if mid.count('.') >= 2 or mid.count('!') >= 2 or mid.count('?') >=2 or mid.count(',') >= 2:
                tmp_mask.append(0)
            elif len(word_tokenize(mid)) - len(list(set(word_tokenize(mid)))) >= 4:
                tmp_mask.append(0)
            else:
                pred_ratio = fuzz.ratio(src, mid) 
                if pred_ratio >= 70:
                    tmp_mask.append(0)
                else:
                    tmp_mask.append(1)
        results.append(tmp_mask)
    return torch.LongTensor(results)


# deductive ----> abductive
def primal(hps, abductive: GPT2LMHeadModel, deductive: GPT2LMHeadModel, inductive, causality, batch, tokenizer, tokenizer_c, fo):

    # induction
    if hps.induction == 'prompt' or hps.induction == 'comet':
        entities = []
    elif hps.induction == 'chain':
        entities = batch[-2]
    else:
        entities = batch[2]
    entities = [e if len(e) > 0 else 'None' for e in entities]
    inductive_out = inductive(batch[:2], entities, deductive, 'deductive')
    # pdb.set_trace()
    effect_mid_ids, dists, index = sample_sequence_continual2(deductive, [inductive_out[-2], inductive_out[-1]], hps.max_len, deductive.config.n_ctx, tokenizer)
    # pdb.set_trace()
    effect_mid = [tokenizer.batch_decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in effect_mid_ids]
    effect_mid = [[process_text(e) for e in es] for es in effect_mid]
    effect_mask = get_mask(batch[0], effect_mid, 'deductive').to(inductive_out[-2].device)

    effect_mid_ids = effect_mid_ids.view(-1, effect_mid_ids.size(-1))
    losses = []
    for i in range(len(effect_mid)*len(effect_mid[0])):
        losses.append(loss_nll(dists[i, :index[i], :], effect_mid_ids[i, :index[i]]))
    
    loss_fw = torch.stack(losses, 0).view(len(batch[0]), -1) # B * N
    loss_bw = compute_reward2(abductive, batch[0], effect_mid, inductive, 'abductive') # B * N

    reward1 = compute_reward1(hps, causality, tokenizer_c, batch[0], effect_mid, 'deductive') # B * N: causality
    reward2 = Variable(loss_bw.data, requires_grad=False) # validation
    if reward2.size(1) != 1:
        reward2 = ((torch.mean(reward2, 1) - reward2.transpose(0, 1)) / torch.std(reward2, 1)).transpose(0, 1)
    reward3 = compute_consine_similarity(hps, batch[0], effect_mid, deductive, 'deductive', tokenizer) # cosine
    reward = hps.alpha * reward1 + (1 - hps.alpha) * reward2 + 0.2 * reward3
    
    # wo causality
    # reward = reward2 + 0.2 * reward3

    # wo cosine
    # reward = hps.alpha * reward1 + (1 - hps.alpha) * reward2
    
    reward = reward * effect_mask

    # json.dump([{'cause': c, 'effects': e, 'loss_fw': lf, 'loos_bw': lb, "reward1": r1, "reward3": r3, "reward": r} for c, e, lf, lb, r1, r3, r in zip(batch[0], effect_mid, loss_fw.cpu().tolist(), loss_bw.cpu().tolist(), reward1.cpu().tolist(), reward3.cpu().tolist(), reward.cpu().tolist())], fo, indent=1)

    lossA = torch.mean(reward * loss_fw)
    lossB = torch.mean(loss_bw * (1 - hps.alpha))

    return lossA, lossB


# abductive ----> deductive
def dual(hps, abductive: GPT2LMHeadModel, deductive: GPT2LMHeadModel, inductive, causality, batch, tokenizer, tokenizer_c, fo):

    # induction
    if hps.induction == 'prompt':
        entities = []
    elif hps.induction == 'chain':
        entities = batch[-1]
    else:
        entities = batch[2]
    entities = [e if len(e) > 0 else 'None' for e in entities]
    inductive_out = inductive([batch[1], batch[0]], entities, abductive, 'abductive')

    cause_mid_ids, dists, index = sample_sequence_continual2(abductive, [inductive_out[-2], inductive_out[-1]], hps.max_len, abductive.config.n_ctx, tokenizer)
    cause_mid = [tokenizer.batch_decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in cause_mid_ids]
    cause_mid = [[process_text(e) for e in es] for es in cause_mid]
    cause_mask = get_mask(cause_mid, batch[0], 'abductive').to(inductive_out[-2].device)

    cause_mid_ids = cause_mid_ids.view(-1, cause_mid_ids.size(-1))
    losses = []
    for i in range(len(cause_mid)*len(cause_mid[0])):
        losses.append(loss_nll(dists[i, :index[i], :], cause_mid_ids[i, :index[i]]))
    loss_fw = torch.stack(losses, 0).view(len(batch[0]), -1) # B * N
    loss_bw = compute_reward2(deductive, cause_mid, batch[0], inductive, 'deductive') # B * N

    reward1 = compute_reward1(hps, causality, tokenizer_c, cause_mid, batch[0], 'abductive') # B * N
    reward2 = Variable(loss_bw.data, requires_grad=False)
    if reward2.size(1) != 1:
        reward2 = ((torch.mean(reward2, 1) - reward2.transpose(0, 1)) / (torch.std(reward2, 1) + 1e-8)).transpose(0, 1)
    reward3 = compute_consine_similarity(hps, cause_mid, batch[0], abductive, 'abductive', tokenizer)
    reward = hps.alpha * reward1 + (1 - hps.alpha) * reward2 + 0.2 * reward3
    
    # wo causality
    # reward = reward2 + 0.2 * reward3

    # wo cosine
    # reward = hps.alpha * reward1 + (1 - hps.alpha) * reward2

    reward = reward * cause_mask

    # json.dump([{'cause': c, 'effects': e, 'loss_fw': lf, 'loos_bw': lb, "reward1": r1, 'reward': r3, "reward": r} for c, e, lf, lb, r1, r3, r in zip(cause_mid, batch[0], loss_fw.cpu().tolist(), loss_bw.cpu().tolist(), reward1.cpu().tolist(), reward3.cpu().tolist(), reward.cpu().tolist())], fo, indent=1)


    lossA = torch.mean(reward * loss_fw)
    lossB = torch.mean(loss_bw * (1 - hps.alpha))

    return lossA, lossB


def retrieve_paths(srcs, dsts, triples):
    relation_dict = {}
    for x, y, z in triples:
        if x not in relation_dict:
            relation_dict[x] = [[y[0], z]]
        else:
            relation_dict[x].append([y[0], z])
    
    chains = [[s] for s in srcs]
    for _ in range(2):
        tmp_chains = []
        for s in chains:
            if s[-1] in relation_dict:
                tails = [t for t in relation_dict[s] if t in dsts][:5]
                tmp_chains += [s + t for t in tails]
        chains = tmp_chains
    chains = [' '.join(chain) for chain in chains[:10]]
    return chains


def induction(sentences):
    matched_concepts = match_mentioned_concepts(sentences)
    res = []
    chains = []

    for concepts in matched_concepts:
        retrieved_entities, _, distances, triples = find_neighbours_frequency(concepts, 2, max_B=100)
        res.append(', '.join(retrieved_entities) if len(retrieved_entities) > 2 else 'None')

        # src_entities = [retrieved_entities[i] for i, d in enumerate(distances) if d == 0]
        # dst_entities = [retrieved_entities[i] for i, d in enumerate(distances) if d != 0]
        # chains.append(retrieve_paths(src_entities, dst_entities, triples))

    return res, chains


def compute_reward1(hps, causality, tokenizer, causes, effects, mode):
    pairs = []
    if mode == 'deductive':
        for i in range(len(causes)):
            pairs += [[causes[i], e] for e in effects[i]]
    else:
        for i in range(len(effects)):
            pairs += [[c, effects[i]] for c in causes[i]]
    
    inputs = tokenizer(pairs, padding=True, return_tensors='pt')
    inputs = [term.cuda(hps.gpu[0]) for term in inputs.values()]
    with torch.no_grad():
        logits = causality(input_ids=inputs[0], attention_mask=inputs[1]).logits # BN * 2
        logits = torch.softmax(logits, 1)
        scores = logits[:, 1].view(len(causes), -1) # B * N
        if scores.size(1) != 1:
            scores = ((scores.transpose(0, 1) - torch.mean(scores, 1)) / (torch.std(scores, 1) + 1e-8)).transpose(0, 1)
    
    scores = Variable(scores.data, requires_grad=False)
    return scores


def compute_reward2(model, causes, effects, inductive, mode):
    inputs, outputs = [], []
    B = len(causes)
    if mode == 'deductive':
        N = len(causes[0])
        for i in range(len(causes)):
            inputs += causes[i]
            outputs += [effects[i] for _ in range(len(causes[i]))]
        entities, _ = induction(inputs)
    else:
        N = len(effects[0])
        for i in range(len(effects)):
            inputs += effects[i]
            outputs += [causes[i] for _ in range(len(effects[i]))]
        entities, _ = induction(inputs)
    
    inductive_out = inductive([inputs, outputs], entities, model, 'deductive')

    logits = model(inputs_embeds=inductive_out[0], attention_mask=inductive_out[1], labels=inductive_out[2]).logits
    logits = logits[..., :-1, :].contiguous() # NB * S * V
    labels = inductive_out[2][..., 1:].contiguous() # NB * S

    losses = []
    for i in range(labels.size(0)):
        losses.append(loss_ce(logits[i], labels[i]))
    losses = torch.stack(losses, 0)
    losses = losses.view(B, N) # B * 3

    return losses


def get_token_embedding(hps, model, input_ids):
    if hps.model_name == 'gpt2':
        token_embed = [model.state_dict()['transformer.wte.weight'][input_id] for input_id in input_ids]
        return token_embed
    else:
        return None


def compute_consine_similarity(hps, causes, effects, model, mode, tokenizer):
    inputs, outputs = [], []
    if mode == 'deductive':
        for c, es in zip(causes, effects):
            inputs += [c]*len(es)
            outputs += es
    else:
        for cs, e in zip(causes, effects):
            inputs += cs
            outputs += [e]*len(cs)
    
    ti = tokenizer(inputs)
    to = tokenizer(outputs)
    with torch.no_grad():
        ti_embed = get_token_embedding(hps, model, ti.input_ids)
        to_embed = get_token_embedding(hps, model, to.input_ids)

        ti_embed = torch.stack([torch.mean(embed, 0) for embed in ti_embed], 0)
        to_embed = torch.stack([torch.mean(embed, 0) for embed in to_embed], 0)

        cosine = torch.abs(torch.cosine_similarity(ti_embed, to_embed))
        # reward = 1.0 - cosine.view(len(causes), -1)
        # reward = Variable(reward.data, requires_grad=False)
        reward = Variable(cosine.view(len(causes), -1).data, requires_grad=False)
        if reward.size(1) != 1:
            reward = ((torch.mean(reward, 1) - reward.transpose(0, 1)) / (torch.std(reward, 1) + 1e-8)).transpose(0, 1)
    return reward

