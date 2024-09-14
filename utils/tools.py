import torch
from transformers import BartTokenizer, GPT2LMHeadModel, GPT2Tokenizer, BertModel, BertTokenizer
import jsonlines
from nltk import bleu
import pdb
from rouge import Rouge
import copy
import random
import sys
from nltk import word_tokenize
from nltk import pos_tag_sents
from nltk.corpus import stopwords
import json
from module.generate import sample_sequence


random.seed(1004)


def read_data(path, portion=1.0):
    fi = jsonlines.open(path, 'r')
    causes, effects = [], []
    for line in fi:
        # hypos.append(line['general_truth'])
        causes.append(line['cause'])
        effects.append(line['effect'])
    number = round(len(causes) * portion)
    return causes[:number], effects[:number]


def get_embedding(hps, model, input_ids):
    if hps.model_name == 'gpt2':
        token_embed = model.state_dict()['transformer.wte.weight'][input_ids]
        return token_embed
    elif hps.model_name == 'bart':
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        token_embed = model.get_encoder().embed_tokens(input_ids) * model.get_encoder(). embed_scale
        return token_embed
    elif hps.model_name == 't5':
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        token_embed = model.encoder.embed_tokens(input_ids)
        return token_embed
    else:
        return None


def read_retrieved_knowledge(path, portion=1.0):
    data = json.load(open(path, 'r'))
    cause_entities, effect_entities, cause_chains, effect_chains = [], [], [], []
    for instance in data:
        cause_entities.append(instance['cause'][0])
        effect_entities.append(instance['effect'][0])
        cause_chains.append(';'.join(instance['cause_chains'][0]))
        effect_chains.append(';'.join(instance['effect_chains'][0]))
    number = round(len(cause_entities) * portion)
    return cause_entities[:number], effect_entities[:number], cause_chains[:number], effect_chains[:number]


def process_text(sent: str):
    if '.' in sent and not sent.startswith('.'):
        sent = sent.split('.')[0] + '.'
    while '\n\n' in sent:
        sent = sent.replace('\n\n', '\n')
    sent = sent.replace('\n', ' ')
    return sent.strip()


def dataloader(data, batch_size):
    num_batch = len(data[0]) // batch_size if len(data[0]) % batch_size == 0 else len(data[0]) // batch_size + 1
    # pdb.set_trace()
    # while True:
    for i in range(num_batch):
        yield [term[i*batch_size: (i+1)*batch_size] for term in data]


def tokenize4gen(text, tokenizer: GPT2Tokenizer):
    outputs = tokenizer(text, padding=True, return_tensors='pt', return_length=True)
    outputs = [torch.LongTensor(outputs[term]) for term in outputs]
    outputs[1:] = outputs[1:][::-1]
    return outputs


def tokenize_gpt2(inputs, outputs, tokenizer):
    inputs_tokenize = tokenizer(inputs)
    outputs_tokenize = tokenizer(outputs)

    true_labels, input_ids, attention_mask, lengths = [], [], [], []

    for i in range(len(inputs)):
        length1 = len(inputs_tokenize.input_ids[i])
        length2 = len(outputs_tokenize.input_ids[i])
        true_labels.append([-100] * length1 + outputs_tokenize.input_ids[i])

        input_ids.append(inputs_tokenize.input_ids[i] + outputs_tokenize.input_ids[i])

        lengths.append(length1 + length2)

    max_len = max(lengths)

    for i, length in enumerate(lengths):
        attention_mask.append([1] * length + [0] * (max_len-length))
        true_labels[i] += [-100] * (max_len-length)
        input_ids[i] += [tokenizer.pad_token_id] * (max_len-length)

    return torch.LongTensor(input_ids), torch.LongTensor(attention_mask), torch.LongTensor(true_labels)



def evaluation_dual(hps, model: GPT2LMHeadModel, dataloader, tokenizer: GPT2Tokenizer, mode='abductive'):
    model.eval()
    bleu_score, rougelr = 0, 0
    outputs = []
    num_instances = 0
    rouge = Rouge()
    with torch.no_grad():
        for batch in dataloader:
            if mode == 'abductive':
                h, premise = batch
            else:
                premise, h = batch
            
            inputs = tokenize4gen(premise, tokenizer)
            if hps.use_gpu:
                inputs = [term.cuda(hps.gpu[0]) for term in inputs]
            inputs[0] = get_embedding(hps, model, inputs[0])
            # generated_ids = model.generate(input_ids=inputs[0],
            #                                attention_mask=inputs[1],
            #                             #    num_beams=hps.beam_size,
            #                                early_stopping=True,
            #                             #    no_repeat_ngram_size=hps.no_repeat_ngram_size,
            #                             #    repetition_penalty=hps.repetition_penalty,
            #                                output_hidden_states=True,
            #                                output_scores=True,
            #                                do_sample=True,
            #                                return_dict_in_generate=True,
            #                                max_length=inputs[0].shape[-1]+hps.max_len,
            #                                pad_token_id=tokenizer.eos_token_id)
            # pdb.set_trace()
            # generated_ids = generated_ids['sequences'][:, inputs[0].shape[-1]:]
            # generated_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

            generated_ids = sample_sequence(hps, model, inputs, hps.max_len, model.config.n_ctx, tokenizer, 1.0, 8, 0, 1.5)
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            generated_text = [process_text(g) for g in generated_text]

            for i in range(len(generated_text)):
                outputs.append({'cause': batch[0][i], 'effect': batch[1][i], 'prediction': generated_text[i]})

                bleu_score += bleu([word_tokenize(h[i])], word_tokenize(generated_text[i]), [0.25, 0.25, 0.25, 0.25])

                try:
                    scores = rouge.get_scores(generated_text[i], h[i])
                    rougel = scores[0]['rouge-l']
                    rougelr += rougel['r']
                except:
                    continue

            num_instances += len(h)
        avg_bleu = bleu_score / num_instances
        rouge_l = rougelr / num_instances

    model.train()
    
    return avg_bleu, rouge_l, outputs

def evaluation_loop(hps, model, inductive, dataloader, tokenizer: GPT2Tokenizer, mode='abductive'):
    model.eval()
    inductive.eval()
    bleu_score = 0
    rougelr = 0
    outputs = []
    num_instances = 0
    rouge = Rouge()
    with torch.no_grad():
        for batch in dataloader:
            if mode == 'abductive':
                h, premise = batch[:2]
                if hps.induction == 'chain':
                    entities = batch[-1]
                else:
                    entities = batch[3]
            else:
                premise, h = batch[:2]
                if hps.induction == 'chain':
                    entities = batch[-2]
                else:
                    entities = batch[2]

            entities = [e if len(e) > 0 else 'None' for e in entities]
            inductive_out = inductive(batch[:2], entities, model, mode)
            
            generated_ids = sample_sequence(hps, model, [inductive_out[-2], inductive_out[-1]], hps.max_len, model.config.n_ctx, tokenizer, 1.0, 8, 0, 1.5)
            generated_text = tokenizer.batch_decode(generated_ids)

            generated_text = [process_text(g) for g in generated_text]

            for i in range(len(generated_text)):
                outputs.append({'cause': batch[0][i], 'effect': batch[1][i], 'prediction': generated_text[i]})

                bleu_score += bleu([word_tokenize(h[i])], word_tokenize(generated_text[i]), [0.25, 0.25, 0.25, 0.25])

                try:
                    scores = rouge.get_scores(generated_text[i], h[i])
                    rougel = scores[0]['rouge-l']
                    rougelr += rougel['r']
                except:
                    continue

            num_instances += len(h)
        avg_bleu = bleu_score / num_instances
        rouge_l = rougelr / num_instances

    model.train()
    inductive.train()
    
    return avg_bleu, rouge_l, outputs


def evaluation_loop_bart(hps, model, inductive, dataloader, tokenizer: BartTokenizer, mode='abductive'):
    model.eval()
    inductive.eval()
    bleu_score = 0
    rougelr = 0
    outputs = []
    num_instances = 0
    rouge = Rouge()
    with torch.no_grad():
        for batch in dataloader:
            if mode == 'abductive':
                h, premise = batch[:2]
                if hps.induction == 'chain':
                    entities = batch[-1]
                else:
                    entities = batch[3]
            else:
                premise, h = batch[:2]
                if hps.induction == 'chain':
                    entities = batch[-2]
                else:
                    entities = batch[2]

            inductive_out = inductive(batch[:2], entities, model, mode)

            generated_ids = model.generate(inputs_embeds=inductive_out[0],
                                           attention_mask=inductive_out[1],
                                        #    num_beams=hps.beam_size,
                                           early_stopping=True,
                                           no_repeat_ngram_size=hps.no_repeat_ngram_size,
                                           repetition_penalty=hps.repetition_penalty,
                                           output_hidden_states=True,
                                           output_scores=True,
                                           return_dict_in_generate=True,
                                           max_length=hps.max_len,
                                           pad_token_id=tokenizer.pad_token_id)
            # pdb.set_trace()
            generated_ids = generated_ids['sequences']
            generated_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            generated_text = [process_text(g) for g in generated_text]

            for i in range(len(generated_text)):
                outputs.append({'cause': batch[0][i], 'effect': batch[1][i], 'prediction': generated_text[i]})

                bleu_score += bleu([word_tokenize(h[i])], word_tokenize(generated_text[i]), [0.25, 0.25, 0.25, 0.25])

                try:
                    scores = rouge.get_scores(generated_text[i], h[i])
                    rougel = scores[0]['rouge-l']
                    rougelr += rougel['r']
                except:
                    continue

            num_instances += len(h)
        avg_bleu = bleu_score / num_instances
        rouge_l = rougelr / num_instances
        # ppl = ppl / num_instances

    model.train()
    inductive.train()
    
    return avg_bleu, rouge_l, outputs



def evaluation_gpt2(hps, model, dataloader, tokenizer: GPT2Tokenizer):
    model.eval()
    outputs = []
    inputs = []
    predictions = []
    references = []
    bleu_score, rougelr = 0, 0
    rouge = Rouge()

    with torch.no_grad():
        for batch in dataloader:
            inputs += [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in batch[0]]
            references += batch[2]
            if hps.use_gpu:
                batch = [term.cuda(hps.gpu[0]) for term in batch[:2]]
            batch[0] = get_embedding(hps, model, batch[0])

            # generated_ids = model.generate(input_ids=batch[0],
            #                             attention_mask=batch[1],
            #                             # num_beams=hps.beam_size,
            #                             early_stopping=True,
            #                             # no_repeat_ngram_size=hps.no_repeat_ngram_size,
            #                             # repetition_penalty=hps.repetition_penalty,
            #                             output_hidden_states=True,
            #                             output_scores=True,
            #                             do_sample=True,
            #                             return_dict_in_generate=True,
            #                             max_length=batch[0].shape[-1]+hps.max_len,
            #                             pad_token_id=tokenizer.eos_token_id)
            
            # generated_ids = generated_ids['sequences'][:, batch[0].shape[-1]:]
            # generated_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

            generated_ids = sample_sequence(hps, model, batch, hps.max_len, model.config.n_ctx, tokenizer, 1, 8, 0, 1.5)
            generated_text = tokenizer.batch_decode(generated_ids)
            generated_text = [process_text(g) for g in generated_text]

            predictions += generated_text

    
    for p, r, i in zip(predictions, references, inputs):
        if hps.mode == 'abductive':
            outputs.append({'effect': i, 'cause': r, 'prediction': p})
        else:
            outputs.append({'cause': i, 'effect': r, 'prediction': p})

        bleu_score += bleu([word_tokenize(r)], word_tokenize(p), [0.25, 0.25, 0.25, 0.25])

        try:
            scores = rouge.get_scores(p, r)
            rougelr += scores[0]['rouge-l']['r']
        except:
            continue
    model.train()
    return bleu_score / len(references), rougelr / len(references), outputs


def filter_pos(pos_tags):
    noun_adj_words = []
    for sentence in pos_tags:
        meaningful_words = []
        tmp_words = []
        pre = 'Random'
        for word in sentence:
            if word[1].startswith(pre):
                tmp_words.append(word[0])
            else:
                if pre in ['NN', 'JJ']:
                    meaningful_words.append(' '.join(tmp_words).lower())
                pre = word[1][:2]
                tmp_words = [word[0]]
        if pre in ['NN', 'JJ']:
            meaningful_words.append(' '.join(tmp_words).lower())

        # noun_adj_words.append([meaningful_words, sentence])
        noun_adj_words.append(meaningful_words)
    return noun_adj_words


def retrieve(entities, entity2id, id2entity, entity2children, entity2parents, hops):
    leaves = copy.deepcopy(entities)
    entities = []
    for _ in range(hops):
        tmp_leaves = []
        random.shuffle(leaves)
        leaves = leaves[:50]
        for entity in leaves:
            try:
                children = entity2children[entity]
            except:
                children = {}
            
            try:
                parents = entity2parents[entity]
            except:
                parents = {}
            
            if len(parents) == len(children) == 0:
                continue

            for r, cs in children.items():
                cs = list(set([id2entity[c] for c in cs]))
                cs = [c for c in cs[:2] if c not in entities]
                
                entities += cs
                tmp_leaves += cs

            for r, cs in parents.items():
                cs = list(set([id2entity[c] for c in cs]))
                cs = [c for c in cs[:2] if c not in entities]
                
                entities += cs
                tmp_leaves += cs
        leaves = tmp_leaves
    random.shuffle(entities)
    return ';'.join(entities[:50])


english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
stop_words = list(stopwords.words("english"))
stop_words += ['man', 'person', 'male', 'female', 
               'human', 'female human', 'male human', 
               'female person', 'male person', 'men',
               'woman', 'women', 'kid', 'child', 'children',
               'boy', 'girl', 'kids', 'boys', 'girls']
stop_words = list(set(stop_words))
indexes = json.load(open('', 'r'))
entity2id, id2entity, entity2children, entity2parents = [indexes[key] for key in indexes]
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']


def induction(sentences):
    sentences = [word_tokenize(s) for s in sentences]
    sentences = [[word for word in words if word not in punctuations+stop_words] for words in sentences]
    pos_tags = pos_tag_sents(sentences)

    sentences = filter_pos(pos_tags)

    entities = [retrieve(sent, entity2id, id2entity, entity2children, entity2parents, 3) for sent in sentences]
    return entities, None


# def retrieve_paths(srcs, dsts, triples):
#     relation_dict = {}
#     for x, y, z in triples:
#         if x not in relation_dict:
#             relation_dict[x] = [[y[0], z]]
#         else:
#             relation_dict[x].append([y[0], z])
    
#     chains = [[s] for s in srcs]
#     for _ in range(2):
#         tmp_chains = []
#         for s in chains:
#             if s[-1] in relation_dict:
#                 tails = [t for t in relation_dict[s] if t in dsts][:5]
#                 tmp_chains += [s + t for t in tails]
#         chains = tmp_chains
#     chains = [' '.join(chain) for chain in chains[:10]]
#     return chains


# def induction(sentences):
#     matched_concepts = match_mentioned_concepts(sentences)
#     res = []
#     chains = []

#     for concepts in matched_concepts:
#         retrieved_entities, _, distances, triples = find_neighbours_frequency(concepts, 2, max_B=100)
#         res.append(';'.join(retrieved_entities))

#         src_entities = [retrieved_entities[i] for i, d in enumerate(distances) if d == 0]
#         dst_entities = [retrieved_entities[i] for i, d in enumerate(distances) if d != 0]
#         chains.append(retrieve_paths(src_entities, dst_entities, triples))

#     return res, chains



