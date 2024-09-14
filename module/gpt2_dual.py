import torch.nn as nn
from transformers import GPT2LMHeadModel
from utils.tools import tokenize4gen, tokenize_gpt2
import pdb
import torch

def get_embedding(hps, model, input_ids):
    if hps.model_name == 'gpt2':
        token_embed = model.state_dict()['transformer.wte.weight'][input_ids]
        return token_embed
    else:
        return None


# deductive ----> abductive
def primal(hps, abductive: GPT2LMHeadModel, deductive: GPT2LMHeadModel, batch, tokenizer):

    for_gen = tokenize4gen(batch[0], tokenizer)
    if hps.use_gpu:
        for_gen = [term.cuda(hps.gpu[0]) for term in for_gen]
    
    # deductive.eval()
    # with torch.no_grad():
    effect_mid_output = deductive.generate(input_ids=for_gen[0],
                                        attention_mask=for_gen[1],
                                        # num_beams=hps.beam_size,
                                        early_stopping=True,
                                        # no_repeat_ngram_size=hps.no_repeat_ngram_size,
                                        # repetition_penalty=hps.repetition_penalty,
                                        # num_return_sequences=hps.multi_sent,
                                        output_hidden_states=True,
                                        output_scores=True,
                                        do_sample=True,
                                        return_dict_in_generate=True,
                                        max_length=for_gen[0].shape[-1]+hps.max_len,
                                        pad_token_id=tokenizer.eos_token_id)
    # deductive.train()

    cause_embeds = torch.stack([h[0][:, -1, :] for h in effect_mid_output.hidden_states[:for_gen[0].size(1)]], 1) # B x S1 x H
    # pdb.set_trace()
    cause_embeds = torch.stack([torch.mean(c[:for_gen[-1][i].item(), :], 0) for i, c in enumerate(cause_embeds)], 0)
    if hps.multi_sent == 1:
        if not hps.e2e:
            effect_mid_ids = effect_mid_output.sequences[:, for_gen[0].shape[1]:]
            effect_mid = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace('\n', '') for g in effect_mid_ids]
        else:
            effect_mid = torch.stack([h[-1][:, -1, :] for h in effect_mid_output.hidden_states], 1) # B x S2 x H
    else:
        if not hps.e2e:
            effect_mid_ids = effect_mid_output['sequences'].view(hps.multi_sent, len(batch[0]), -1)[:, :, for_gen[0].shape[1]:]
            effect_mid = [[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in gs] for gs in effect_mid_ids]
        else:
            effect_mid = torch.stack([h[-1][:, -1, :] for h in effect_mid_output.hidden_states], 1)
            effect_mid = effect_mid.view(hps.multi_sent, len(batch[0]), -1)

    input_ids, attention_mask, true_labels = tokenize_gpt2(batch[0], batch[1], tokenizer)
    if hps.use_gpu:
        input_ids = input_ids.cuda(hps.gpu[0])
        attention_mask = attention_mask.cuda(hps.gpu[0])
        true_labels = true_labels.cuda(hps.gpu[0])
    
    loss1 = deductive(input_ids=input_ids, attention_mask=attention_mask, labels=true_labels).loss

    if hps.multi_sent == 1:
        if not hps.e2e:
            input_ids, attention_mask, true_labels = tokenize_gpt2(effect_mid, batch[0], tokenizer)
            if hps.use_gpu:
                input_ids = input_ids.cuda(hps.gpu[0])
                attention_mask = attention_mask.cuda(hps.gpu[0])
                true_labels = true_labels.cuda(hps.gpu[0])
            loss2 = abductive(input_ids=input_ids, attention_mask=attention_mask, labels=true_labels).loss
        else:
            cause_embed = get_embedding(hps, abductive, for_gen[0])
            input_embed = torch.cat([effect_mid, cause_embed], 1)
            labels_1 = (torch.zeros_like(effect_mid_output.sequences[:, for_gen[0].shape[1]:])-100).cuda(hps.gpu[0])
            labels_2 = torch.where(for_gen[0]==tokenizer.pad_token_id, -100, for_gen[0])
            labels = torch.cat([labels_1, labels_2], 1)
            attention_mask_1 = torch.ones_like(effect_mid_output.sequences[:, for_gen[0].shape[1]:]).cuda(hps.gpu[0])
            attention_mask_2 = torch.where(for_gen[0]==tokenizer.pad_token_id, 0, torch.ones_like(for_gen[0]))
            attention_mask = torch.cat([attention_mask_1, attention_mask_2], 1)
            loss2 = abductive(inputs_embeds=input_embed, attention_mask=attention_mask, labels=labels).loss
    else:
        loss2 = []
        if not hps.e2e:
            for i in range(len(effect_mid)):
                input_ids, attention_mask, true_labels = tokenize_gpt2(effect_mid[i], batch[0], tokenizer)
                if hps.use_gpu:
                    input_ids = input_ids.cuda(hps.gpu[0])
                    attention_mask = attention_mask.cuda(hps.gpu[0])
                    true_labels = true_labels.cuda(hps.gpu[0])
                loss2.append(abductive(input_ids=input_ids, attention_mask=attention_mask, labels=true_labels).loss)
        else:
            cause_embed = get_embedding(hps, abductive, for_gen[0])
            labels_2 = torch.where(for_gen[0]==tokenizer.pad_token_id, -100, for_gen)
            attention_mask_2 = torch.where(for_gen[0]==tokenizer.pad_token_id, 0, torch.ones_like(for_gen[0]))
            for i in range(len(effect_mid)):
                input_embed = torch.cat([effect_mid[i], cause_embed], 1)
                labels_1 = (torch.zeros(effect_mid[i].size(0), effect_mid[i].size(1))-100).cuda(hps.gpu[0])
                attention_mask_1 = torch.ones(effect_mid[i].size(0), effect_mid[i].size(1)).cuda(hps.gpu[0])
                labels = torch.cat([labels_1, labels_2], 1)
                attention_mask = torch.cat([attention_mask_1, attention_mask_2], 1)
                loss2.append(abductive(inputs_embeds=input_embed, attention_mask=attention_mask, labels=labels).loss)
        loss2 = torch.mean(torch.stack(loss2, 0))
    # pdb.set_trace()
    # loss3 = cause_embeds @ torch.mean(effect_mid, 1).transpose(0, 1) # B * B
    # loss3 = torch.abs(torch.mean(torch.diag(loss3))) - 0.2
    # cos_sim = torch.abs(torch.cosine_similarity(cause_embeds, torch.mean(effect_mid, 1)))
    # loss3 = torch.mean(cos_sim)
    
    return loss1, loss2


# abductive ----> deductive
def dual(hps, abductive: GPT2LMHeadModel, deductive: GPT2LMHeadModel, batch, tokenizer):

    for_gen = tokenize4gen(batch[1], tokenizer)
    if hps.use_gpu:
        for_gen = [term.cuda(hps.gpu[0]) for term in for_gen]

    # abductive.eval()
    # with torch.no_grad():
    cause_mid_output = abductive.generate(input_ids=for_gen[0],
                                        # attention_mask=for_gen[1],
                                        # num_beams=hps.beam_size,
                                        early_stopping=True,
                                        no_repeat_ngram_size=hps.no_repeat_ngram_size,
                                        repetition_penalty=hps.repetition_penalty,
                                        num_return_sequences=hps.multi_sent,
                                        output_hidden_states=True,
                                        output_scores=True,
                                        do_sample=True,
                                        return_dict_in_generate=True,
                                        max_length=for_gen[0].shape[-1]+hps.max_len,
                                        pad_token_id=tokenizer.eos_token_id)
    # abductive.train()
    effect_embeds = torch.stack([h[0][:, -1, :] for h in cause_mid_output.hidden_states[:for_gen[0].size(1)]], 1)
    effect_embeds = torch.stack([torch.mean(c[:for_gen[-1][i], :], 0) for i, c in enumerate(effect_embeds)], 0)
    if hps.multi_sent == 1:
        if not hps.e2e:
            cause_mid_ids = cause_mid_output['sequences'][:, for_gen[0].shape[1]:]
            cause_mid = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace('\n', '') for g in cause_mid_ids]
        else:
            cause_mid = torch.stack([h[-1][:, -1, :] for h in cause_mid_output.hidden_states], 1)
    else:
        if not hps.e2e:
            cause_mid_ids = cause_mid_output['sequences'].view(hps.multi_sent, len(batch[0]), -1)[:, :, for_gen[0].shape[1]:]
            cause_mid = [[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in gs] for gs in cause_mid_ids]
        else:
            cause_mid = torch.stack([h[-1][:, -1, :] for h in cause_mid_output.hidden_states], 1)
            cause_mid = cause_mid.view(hps.multi_sent, len(batch[0]), -1)

    input_ids, attention_mask, true_labels = tokenize_gpt2(batch[1], batch[0], tokenizer)
    if hps.use_gpu:
        input_ids = input_ids.cuda(hps.gpu[0])
        attention_mask = attention_mask.cuda(hps.gpu[0])
        true_labels = true_labels.cuda(hps.gpu[0])
    loss1 = abductive(input_ids=input_ids, attention_mask=attention_mask, labels=true_labels).loss

    if hps.multi_sent == 1:
        if not hps.e2e:
            input_ids, attention_mask, true_labels = tokenize_gpt2(cause_mid, batch[1], tokenizer)
            if hps.use_gpu:
                input_ids = input_ids.cuda(hps.gpu[0])
                attention_mask = attention_mask.cuda(hps.gpu[0])
                true_labels = true_labels.cuda(hps.gpu[0])
            loss2 = deductive(input_ids=input_ids, attention_mask=attention_mask, labels=true_labels).loss
        else:
            effect_embed = get_embedding(hps, deductive, for_gen[0])
            input_embed = torch.cat([cause_mid, effect_embed], 1)
            labels_1 = (torch.zeros_like(cause_mid_output.sequences[:, for_gen[0].shape[1]:])-100).cuda(hps.gpu[0])
            labels_2 = torch.where(for_gen[0]==tokenizer.pad_token_id, -100, for_gen[0])
            labels = torch.cat([labels_1, labels_2], 1)
            attention_mask_1 = torch.ones_like(cause_mid_output.sequences[:, for_gen[0].shape[1]:]).cuda(hps.gpu[0])
            attention_mask_2 = torch.where(for_gen[0]==tokenizer.pad_token_id, 0, torch.ones_like(for_gen[0]))
            attention_mask = torch.cat([attention_mask_1, attention_mask_2], 1)
            loss2 = deductive(inputs_embeds=input_embed, attention_mask=attention_mask, labels=labels).loss
    else:
        loss2 = []
        if not hps.e2e:
            for i in range(len(cause_mid)):
                input_ids, attention_mask, true_labels = tokenize_gpt2(cause_mid[i], batch[1], tokenizer)
                if hps.use_gpu:
                    input_ids = input_ids.cuda(hps.gpu[0])
                    attention_mask = attention_mask.cuda(hps.gpu[0])
                    true_labels = true_labels.cuda(hps.gpu[0])
                loss2.append(deductive(input_ids=input_ids, attention_mask=attention_mask, labels=true_labels).loss)
        else:
            effect_embed = get_embedding(hps, abductive, for_gen[0])
            labels_2 = torch.where(for_gen[0]==tokenizer.pad_token_id, -100, for_gen[0])
            attention_mask_2 = torch.where(for_gen[0]==tokenizer.pad_token_id, 0, torch.ones_like(for_gen[0]))
            for i in range(len(cause_mid)):
                input_embed = torch.cat([cause_mid[i], effect_embed], 1)
                labels_1 = (torch.zeros(cause_mid[i].size(0), cause_mid[i].size(1))-100).cuda(hps.gpu[0])
                attention_mask_1 = torch.ones(cause_mid[i].size(0), cause_mid[i].size(1)).cuda(hps.gpu[0])
                labels = torch.cat([labels_1, labels_2], 1)
                attention_mask = torch.cat([attention_mask_1, attention_mask_2], 1)
                loss2.append(deductive(inputs_embeds=input_embed, attention_mask=attention_mask, labels=labels).loss)
        loss2 = torch.mean(torch.stack(loss2, 0))
    
    # loss3 = effect_embeds @ torch.mean(cause_mid, 1).transpose(0, 1) # B * B
    # loss3 = torch.abs(torch.mean(torch.diag(loss3))) - 0.2
    # cos_sim = torch.abs(torch.cosine_similarity(effect_embeds, torch.mean(cause_mid, 1)))
    # loss3 = torch.mean(cos_sim)

    return loss1, loss2









