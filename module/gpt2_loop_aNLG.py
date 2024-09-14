from transformers import GPT2LMHeadModel
from utils.rule_retrieval import find_neighbours_frequency, match_mentioned_concepts
from .generate import sample_sequence
import pdb



def process_text(sent: str):
    if '.' in sent and not sent.startswith('.'):
        sent = sent.split('.')[0] + '.'
    while '\n\n' in sent:
        sent = sent.replace('\n\n', '\n')
    sent = sent.replace('\n', ' ')
    return sent.strip()


# deductive ----> abductive
def primal(hps, abductive: GPT2LMHeadModel, deductive: GPT2LMHeadModel, inductive, batch, tokenizer):

    # induction
    if hps.induction == 'chain':
        entities = [p+';'+c for p, c in zip(batch[6], batch[7])]
    else:
        entities = [p+';'+c for p, c in zip(batch[3], batch[4])]

    entities = [e if len(e) > 0 else 'None' for e in entities]

    inductive_out = inductive([[p+c for p, c in zip(batch[0], batch[1])], batch[2]], entities, deductive, 'deductive')

    effect_mid_ids = sample_sequence(hps, deductive, [inductive_out[-2], inductive_out[-1]], hps.max_len, deductive.config.n_ctx, tokenizer, 1.0, 8, 0, 1.5)
    effect_mid = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in effect_mid_ids]
    effect_mid = [process_text(e) for e in effect_mid]

    loss1 = deductive(inputs_embeds=inductive_out[0], attention_mask=inductive_out[1], labels=inductive_out[2]).loss

    if hps.induction == 'chain':
        _, entities = induction(effect_mid)
        entities = [p+';'+';'.join(e) for p, e in zip(batch[6], entities)]
    else:
        entities, _ = induction(effect_mid)
        entities = [p+', '+e for p, e in zip(batch[3], entities)]

    entities = [e if len(e) > 0 else 'None' for e in entities]   
    inductive_out = inductive([batch[1], [p+e for p, e in zip(batch[0], effect_mid)]], entities, abductive, 'abductive')

    loss2 = abductive(inputs_embeds=inductive_out[0], attention_mask=inductive_out[1], labels=inductive_out[2]).loss

    return loss1, loss2


# abductive ----> deductive
def dual(hps, abductive: GPT2LMHeadModel, deductive: GPT2LMHeadModel, inductive, batch, tokenizer):

    if hps.induction == 'chain':
        entities = [p+';'+e for p, e in zip(batch[6], batch[8])]
    else:
        entities = [p+';'+e for p, e in zip(batch[3], batch[5])]
    entities = [e if len(e) > 0 else 'None' for e in entities]
    inductive_out = inductive([batch[1], [p+e for p, e in zip(batch[0], batch[2])]], entities, abductive, 'abductive')

    cause_mid_ids = sample_sequence(hps, abductive, [inductive_out[-2], inductive_out[-1]], hps.max_len, abductive.config.n_ctx, tokenizer, 1.0, 8, 0, 1.5)
    cause_mid = tokenizer.batch_decode(cause_mid_ids)
    cause_mid = [process_text(e) for e in cause_mid]
    
    loss1 = abductive(inputs_embeds=inductive_out[0], attention_mask=inductive_out[1], labels=inductive_out[2]).loss

    if hps.induction == 'chain':
        _, entities = induction(cause_mid)
        entities = [p+';'+ ';'.join(c) for p, c in zip(batch[6], entities)]
    else:
        entities, _ = induction(cause_mid)
        entities = [p+';'+ c for p, c in zip(batch[3], entities)]

    entities = [e if len(e) > 0 else 'None' for e in entities]
    inductive_out = inductive([[p+c for p, c in zip(batch[0], cause_mid)], batch[2]], entities, deductive, 'deductive')

    loss2 = deductive(inputs_embeds=inductive_out[0], attention_mask=inductive_out[1], labels=inductive_out[2]).loss

    return loss1, loss2


def retrieve_paths(srcs, dsts, triples):
    relation_dict = {}
    for x, y, z in triples:
        if x not in relation_dict:
            relation_dict[x] = [[y, z]]
        else:
            relation_dict[x].append([y, z])
    
    chains = [[s] for s in srcs]
    for _ in range(2):
        tmp_chains = []
        for s in chains:
            if s[-1] in relation_dict:
                tails = [t for t in relation_dict[s[-1]]][0:1]
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
        res.append(', '.join(retrieved_entities))

        src_entities = [retrieved_entities[i] for i, d in enumerate(distances[:100]) if d == 0]
        dst_entities = [retrieved_entities[i] for i, d in enumerate(distances[:100]) if d != 0]
        chains.append(retrieve_paths(src_entities, dst_entities, triples))

    return res, chains

