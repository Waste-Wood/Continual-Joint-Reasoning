import json
from transformers import BartTokenizer, GPT2Tokenizer
from rouge import Rouge
import torch
from nltk import bleu
from nltk.tokenize import word_tokenize
from module.generate import sample_sequence


def process_text(sent: str):
    if '.' in sent and not sent.startswith('.'):
        sent = sent.split('.')[0] + '.'
    while '\n\n' in sent:
        sent = sent.replace('\n\n', '\n')
    sent = sent.replace('\n', ' ')
    return sent.strip()


def read_data(path):
    data = json.load(open(path, 'r'))

    premises, causes, effects = [], [], []
    p_entities, c_entities, e_entities = [], [], []
    for e in data:
        for i, h in enumerate(e['hs']):
            premises.append(e['o1'])
            causes.append(h)
            effects.append(e['o2'])
            p_entities.append(e['o1_entities'][0])
            c_entities.append(e['hs_entities'][0][i])
            e_entities.append(e['o2_entities'][0])
    return premises, causes, effects, p_entities, c_entities, e_entities


def evaluation_loop_bart(hps, model, inductive, dataloader, tokenizer: BartTokenizer, mode='abductive'):
    model.eval()
    inductive.eval()
    bleu_score = 0
    rougelr = 0
    results = []
    num_instances = 0
    rouge = Rouge()
    with torch.no_grad():
        for batch in dataloader:
            premises, hypotheses, observations = batch[:3]
            if mode == 'abductive':
                inputs = [p+o for p, o in zip(premises, observations)]
                outputs = hypotheses
                if hps.induction == 'chain':
                    entities = [p+o for p, o in zip(batch[6], batch[8])]
                else:
                    entities = [p+o for p, o in zip(batch[3], batch[5])]
            else:
                inputs = [p+h for p, h in zip(premises, hypotheses)]
                outputs = observations
                if hps.induction == 'chain':
                    entities = [p+o for p, o in zip(batch[6], batch[7])]
                else:
                    entities = [p+o for p, o in zip(batch[3], batch[4])]

            inductive_out = inductive([inputs, outputs], entities, model, 'deductive')

            generated_ids = model.generate(inputs_embeds=inductive_out[0],
                                           attention_mask=inductive_out[1],
                                           early_stopping=True,
                                           no_repeat_ngram_size=hps.no_repeat_ngram_size,
                                           repetition_penalty=hps.repetition_penalty,
                                           return_dict_in_generate=True,
                                           max_length=hps.max_len,
                                           pad_token_id=tokenizer.pad_token_id)

            generated_ids = generated_ids['sequences']
            generated_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            generated_text = [process_text(g) for g in generated_text]

            for i in range(len(generated_text)):
                results.append({'premise': batch[0][i], 'cause': batch[1][i], 'effect': batch[2][i], 'prediction': generated_text[i]})

                bleu_score += bleu([word_tokenize(outputs[i])], word_tokenize(generated_text[i]), [0.25, 0.25, 0.25, 0.25])

                try:
                    scores = rouge.get_scores(generated_text[i], outputs[i])
                    rougel = scores[0]['rouge-l']
                    rougelr += rougel['r']
                except:
                    continue

            num_instances += len(hypotheses)
        avg_bleu = bleu_score / num_instances
        rouge_l = rougelr / num_instances

    model.train()
    inductive.train()
    
    return avg_bleu, rouge_l, results


def evaluation_loop_gpt2(hps, model, inductive, dataloader, tokenizer: GPT2Tokenizer, mode='abductive'):
    model.eval()
    inductive.eval()
    bleu_score = 0
    rougelr = 0
    results = []
    num_instances = 0
    rouge = Rouge()
    with torch.no_grad():
        for batch in dataloader:
            premises, hypotheses, observations = batch[:3]
            if mode == 'abductive':
                inputs = [p+o for p, o in zip(premises, observations)]
                outputs = hypotheses
                if hps.induction == 'chain':
                    entities = [p+o for p, o in zip(batch[6], batch[8])]
                else:
                    entities = [p+o for p, o in zip(batch[3], batch[5])]
            else:
                inputs = [p+h for p, h in zip(premises, hypotheses)]
                outputs = observations
                if hps.induction == 'chain':
                    entities = [p+o for p, o in zip(batch[6], batch[7])]
                else:
                    entities = [p+o for p, o in zip(batch[3], batch[4])]

            entities = [e if len(e) > 0 else 'None' for e in entities]
            inductive_out = inductive([inputs, outputs], entities, model, 'deductive')
            
            generated_ids = sample_sequence(hps, model, [inductive_out[-2], inductive_out[-1]], hps.max_len, model.config.n_ctx, tokenizer, 1.0, 8, 0, 1.5)
            generated_text = tokenizer.batch_decode(generated_ids)

            generated_text = [process_text(g) for g in generated_text]

            for i in range(len(generated_text)):
                results.append({'premise': batch[0][i], 'cause': batch[1][i], 'effect': batch[2][i], 'prediction': generated_text[i]})

                bleu_score += bleu([word_tokenize(outputs[i])], word_tokenize(generated_text[i]), [0.25, 0.25, 0.25, 0.25])

                try:
                    scores = rouge.get_scores(generated_text[i], outputs[i])
                    rougel = scores[0]['rouge-l']
                    rougelr += rougel['r']
                except:
                    continue

            num_instances += len(hypotheses)
        avg_bleu = bleu_score / num_instances
        rouge_l = rougelr / num_instances

    model.train()
    inductive.train()
    
    return avg_bleu, rouge_l, results


