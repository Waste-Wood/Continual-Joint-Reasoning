import torch.nn as nn
from transformers import GPT2LMHeadModel
from utils.tools import tokenize4gen, tokenize_gpt2, filter_pos, retrieve
from nltk import word_tokenize
from nltk import pos_tag_sents
from nltk.corpus import stopwords
import pdb
import torch
import json
from utils.rule_retrieval import find_neighbours_frequency, match_mentioned_concepts
from .generate import sample_sequence


# english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
# stop_words = list(stopwords.words("english"))
# stop_words += ['man', 'person', 'male', 'female', 
#                'human', 'female human', 'male human', 
#                'female person', 'male person', 'men',
#                'woman', 'women', 'kid', 'child', 'children',
#                'boy', 'girl', 'kids', 'boys', 'girls']
# stop_words = list(set(stop_words))


def process_text(sent: str):
    if '.' in sent and not sent.startswith('.'):
        sent = sent.split('.')[0] + '.'
    while '\n\n' in sent:
        sent = sent.replace('\n\n', '\n')
    sent = sent.replace('\n', ' ')
    return sent.strip()


# deductive ----> abductive
def primal(hps, abductive: GPT2LMHeadModel, deductive: GPT2LMHeadModel, inductive_de, batch, tokenizer):

    # induction
    if hps.induction == 'prompt' or hps.induction == 'comet':
        entities = []
    elif hps.induction == 'chain':
        entities = batch[-2]
    else:
        # entities, _ = induction(batch[0])
        entities = batch[2]
    entities = [e if len(e) > 0 else 'None' for e in entities]
    inductive_out = inductive_de(batch[:2], entities, deductive, 'deductive')

    # pdb.set_trace()

    # effect_mid_output = deductive.generate(input_ids=inductive_out[-2],
    #                                     attention_mask=inductive_out[-1],
    #                                     num_beams=hps.beam_size,
    #                                     early_stopping=True,
    #                                     no_repeat_ngram_size=hps.no_repeat_ngram_size,
    #                                     repetition_penalty=hps.repetition_penalty,
    #                                     num_return_sequences=1,
    #                                     output_hidden_states=True,
    #                                     output_scores=True,
    #                                     return_dict_in_generate=True,
    #                                     max_length=inductive_out[-2].shape[-1]+hps.max_len,
    #                                     pad_token_id=tokenizer.eos_token_id)

    # effect_mid_ids = effect_mid_output['sequences'][:, inductive_out[-2].shape[1]:]
    # effect_mid = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in effect_mid_ids]

    effect_mid_ids = sample_sequence(hps, deductive, [inductive_out[-2], inductive_out[-1]], hps.max_len, deductive.config.n_ctx, tokenizer, 1.0, 8, 0, 1.5)
    effect_mid = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in effect_mid_ids]
    effect_mid = [process_text(e) for e in effect_mid]

    # pdb.set_trace()
    # effect_mid_ids = effect_mid_output['sequences'].view(3, hps.batch_size, -1)[:, :, for_gen[0].shape[1]:]
    # effect_mid = [[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in gs] for gs in effect_mid_ids]

    # input_ids, attention_mask, true_labels = tokenize_gpt2(cause, batch[1], tokenizer)
    # if hps.use_gpu:
    #     if input_ids.size(1) >= 480:
    #         input_ids = input_ids[:, -350:].cuda(hps.gpu[0])
    #         attention_mask = attention_mask[:, -350:].cuda(hps.gpu[0])
    #         true_labels = true_labels[:, -350:].cuda(hps.gpu[0])
    #     else:
    #         input_ids = input_ids.cuda(hps.gpu[0])
    #         attention_mask = attention_mask.cuda(hps.gpu[0])
    #         true_labels = true_labels.cuda(hps.gpu[0])
    
    loss1 = deductive(inputs_embeds=inductive_out[0], attention_mask=inductive_out[1], labels=inductive_out[2]).loss

    # loss2 = []
    # for i in range(len(effect_mid)):
    #     input_ids, attention_mask, true_labels = tokenize_gpt2(effect_mid[i], batch[0], tokenizer)
    #     if hps.use_gpu:
    #         input_ids = input_ids.cuda(0)
    #         attention_mask = attention_mask.cuda(0)
    #         true_labels = true_labels.cuda(0)
    #     loss2.append(abductive(input_ids=input_ids, attention_mask=attention_mask, labels=true_labels).loss)
    # loss2 = torch.mean(torch.stack(loss2, 0))

    # induction
    if hps.induction == 'chain':
        _, entities = induction(effect_mid)
        entities = [';'.join(e) for e in entities]
    else:
        entities, _ = induction(effect_mid)
    entities = [e if len(e) > 0 else 'None' for e in entities]
    inductive_out = inductive_de([batch[0], effect_mid], entities, abductive, 'abductive')
    # effect = [entity+'.'+sent for entity, sent in zip(entities, effect_mid)]
    # input_ids, attention_mask, true_labels = tokenize_gpt2(effect, batch[0], tokenizer)
    # if hps.use_gpu:
    #     if input_ids.size(1) >= 480:
    #         input_ids = input_ids[:, -350:].cuda(hps.gpu[0])
    #         attention_mask = attention_mask[:, -350:].cuda(hps.gpu[0])
    #         true_labels = true_labels[:, -350:].cuda(hps.gpu[0])
    #     else:
    #         input_ids = input_ids.cuda(hps.gpu[0])
    #         attention_mask = attention_mask.cuda(hps.gpu[0])
    #         true_labels = true_labels.cuda(hps.gpu[0])
    loss2 = abductive(inputs_embeds=inductive_out[0], attention_mask=inductive_out[1], labels=inductive_out[2]).loss

    return loss1, loss2


# abductive ----> deductive
def dual(hps, abductive: GPT2LMHeadModel, deductive: GPT2LMHeadModel, inductive_de, batch, tokenizer):

    # induction
    if hps.induction == 'chain':
        entities = batch[-1]
    else:
        # entities, _ = induction(batch[1])
        entities = batch[3]
    entities = [e if len(e) > 0 else 'None' for e in entities]
    inductive_out = inductive_de(batch[:2], entities, abductive, 'abductive')

    # for_gen = tokenize4gen(effect, tokenizer)
    # if hps.use_gpu:
    #     if for_gen[0].size(1) >= 480:
    #         for_gen = [term[:, -350:].cuda(hps.gpu[0]) for term in for_gen]
    #     else:
    #         for_gen = [term.cuda(hps.gpu[0]) for term in for_gen]

    # abductive.eval()
    # with torch.no_grad():
    #     cause_mid_output = abductive.generate(input_ids=for_gen[0],
    #                                         attention_mask=for_gen[1],
    #                                         num_beams=hps.beam_size,
    #                                         early_stopping=True,
    #                                         no_repeat_ngram_size=hps.no_repeat_ngram_size,
    #                                         repetition_penalty=hps.repetition_penalty,
    #                                         num_return_sequences=1,
    #                                         output_hidden_states=True,
    #                                         output_scores=True,
    #                                         do_sample=True,
    #                                         return_dict_in_generate=True,
    #                                         max_length=for_gen[0].shape[-1]+hps.max_len,
    #                                         pad_token_id=tokenizer.eos_token_id)
    # deductive.train()

    # cause_mid_ids = cause_mid_output['sequences'][:, for_gen[0].shape[1]:]
    # cause_mid = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in cause_mid_ids]
    # cause_mid_ids = cause_mid_output['sequences'].view(3, hps.batch_size, -1)[:, :, for_gen[0].shape[1]:]
    # cause_mid = [[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in gs] for gs in cause_mid_ids]

    cause_mid_ids = sample_sequence(hps, abductive, [inductive_out[-2], inductive_out[-1]], hps.max_len, abductive.config.n_ctx, tokenizer, 1.0, 8, 0, 1.5)
    cause_mid = tokenizer.batch_decode(cause_mid_ids)
    cause_mid = [process_text(e) for e in cause_mid]

    # input_ids, attention_mask, true_labels = tokenize_gpt2(effect, batch[0], tokenizer)
    # if hps.use_gpu:
    #     if input_ids.size(1) >= 480:
    #         input_ids = input_ids[:, -350:].cuda(hps.gpu[0])
    #         attention_mask = attention_mask[:, -350:].cuda(hps.gpu[0])
    #         true_labels = true_labels[:, -350:].cuda(hps.gpu[0])
    #     else:
    #         input_ids = input_ids.cuda(hps.gpu[0])
    #         attention_mask = attention_mask.cuda(hps.gpu[0])
    #         true_labels = true_labels.cuda(hps.gpu[0])
    
    # loss1 = abductive(input_ids=input_ids, attention_mask=attention_mask, labels=true_labels).loss
    loss1 = abductive(inputs_embeds=inductive_out[0], attention_mask=inductive_out[1], labels=inductive_out[2]).loss


    # multiple sentences
    # loss2 = []
    # for i in range(len(cause_mid)):
    #     input_ids, attention_mask, true_labels = tokenize_gpt2(cause_mid[i], batch[1], tokenizer)
    #     if hps.use_gpu:
    #         input_ids = input_ids.cuda(0)
    #         attention_mask = attention_mask.cuda(0)
    #         true_labels = true_labels.cuda(0)
    #     loss2.append(deductive(input_ids=input_ids, attention_mask=attention_mask, labels=true_labels).loss)
    # loss2 = torch.mean(torch.stack(loss2, 0))

    # induction
    if hps.induction == 'chain':
        _, entities = induction(cause_mid)
        entities = [';'.join(e) for e in entities]
    else:
        entities, _ = induction(cause_mid)
    entities = [e if len(e) > 0 else 'None' for e in entities]
    # cause = [entity+'.'+sent for entity, sent in zip(entities, cause_mid)]
    inductive_out = inductive_de([cause_mid, batch[1]], entities, deductive, 'deductive')
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

        # src_entities = [retrieved_entities[i] for i, d in enumerate(distances[:100]) if d == 0]
        # dst_entities = [retrieved_entities[i] for i, d in enumerate(distances[:100]) if d != 0]
        # chains.append(retrieve_paths(src_entities, dst_entities, triples))

    return res, chains

