import torch
from transformers import BartForConditionalGeneration
import pdb
from utils.rule_retrieval import find_neighbours_frequency, match_mentioned_concepts


def process_text(sent: str):
    if '.' in sent and not sent.startswith('.'):
        sent = sent.split('.')[0] + '.'
    while '\n\n' in sent:
        sent = sent.replace('\n\n', '\n')
    sent = sent.replace('\n', ' ')
    return sent.strip()


# deductive ----> abductive
def primal(hps, abductive: BartForConditionalGeneration, deductive: BartForConditionalGeneration, inductive_de, inductive_ab, batch, tokenizer):

    # induction
    if hps.induction == 'chain':
        entities = batch[-2]
    else:
        entities = batch[2]

    entities = [e if len(e) > 0 else 'None' for e in entities]

    inductive_out = inductive_de(batch[:2], entities, deductive, 'deductive')

    effect_mid_output = deductive.generate(inputs_embeds=inductive_out[0],
                                        attention_mask=inductive_out[1],
                                        do_sample=True,
                                        early_stopping=True,
                                        no_repeat_ngram_size=hps.no_repeat_ngram_size,
                                        repetition_penalty=hps.repetition_penalty,
                                        num_return_sequences=1,
                                        output_hidden_states=True,
                                        return_dict_in_generate=True,
                                        max_length=hps.max_len,
                                        pad_token_id=tokenizer.pad_token_id)

    effect_mid_ids = effect_mid_output['sequences']
    effect_mid = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in effect_mid_ids]
    effect_mid = [process_text(e) for e in effect_mid]

    # e2e training
    effects_hidden = torch.cat([h[-1][:len(batch[0]), :, :] for h in effect_mid_output['decoder_hidden_states']], 1)
    attention_mask = torch.ones(effects_hidden.size(0), effects_hidden.size(1)).cuda(hps.gpu[0])
    
    loss1 = deductive(inputs_embeds=inductive_out[0], attention_mask=inductive_out[1], labels=inductive_out[2]).loss

    # induction
    if hps.induction == 'chain':
        _, entities = induction(effect_mid)
        entities = [';'.join(e) for e in entities]
    else:
        entities, _ = induction(effect_mid)
    entities = [e if len(e) > 0 else 'None' for e in entities]
    inductive_out = inductive_ab([batch[0], effects_hidden], entities, abductive, 'abductive', embeds=True, atten_mask=attention_mask)
    loss2 = abductive(inputs_embeds=inductive_out[0], attention_mask=inductive_out[1], labels=inductive_out[2]).loss

    return loss1, loss2


# abductive ----> deductive
def dual(hps, abductive: BartForConditionalGeneration, deductive: BartForConditionalGeneration, inductive_de, inductive_ab, batch, tokenizer):

    # induction
    if hps.induction == 'chain':
        entities = batch[-1]
    else:
        entities = batch[3]
    entities = [e if len(e) > 0 else 'None' for e in entities]
    inductive_out = inductive_ab(batch[:2], entities, abductive, 'abductive')

    cause_mid_output = abductive.generate(inputs_embeds=inductive_out[0],
                                        attention_mask=inductive_out[1],
                                        # num_beams=hps.beam_size,
                                        early_stopping=True,
                                        no_repeat_ngram_size=hps.no_repeat_ngram_size,
                                        repetition_penalty=hps.repetition_penalty,
                                        num_return_sequences=1,
                                        output_hidden_states=True,
                                        output_scores=True,
                                        do_sample=True,
                                        return_dict_in_generate=True,
                                        max_length=hps.max_len,
                                        pad_token_id=tokenizer.pad_token_id)

    cause_mid_ids = cause_mid_output['sequences']
    cause_mid = tokenizer.batch_decode(cause_mid_ids,skip_special_tokens=True, clean_up_tokenization_spaces=True)
    cause_mid = [process_text(e) for e in cause_mid]

    # e2e training
    causes_hidden = torch.cat([h[-1][:len(batch[0]), :, :] for h in cause_mid_output['decoder_hidden_states']], 1)
    attention_mask = torch.ones(causes_hidden.size(0), causes_hidden.size(1)).cuda(hps.gpu[0])

    loss1 = abductive(inputs_embeds=inductive_out[0], attention_mask=inductive_out[1], labels=inductive_out[2]).loss

    if hps.induction == 'chain':
        _, entities = induction(cause_mid)
        entities = [';'.join(e) for e in entities]
    else:
        entities, _ = induction(cause_mid)

    entities = [e if len(e) > 0 else 'None' for e in entities]
    inductive_out = inductive_de([causes_hidden, batch[1]], entities, deductive, 'deductive', embeds=True, atten_mask=attention_mask)
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

        # src_entities = [retrieved_entities[i] for i, d in enumerate(distances[:len(retrieved_entities)]) if d == 0]
        # dst_entities = [retrieved_entities[i] for i, d in enumerate(distances[:len(retrieved_entities)]) if d != 0]
        # chains.append(retrieve_paths(src_entities, dst_entities, triples))

    return res, chains

