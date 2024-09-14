import torch
import pdb
from transformers import GPT2Tokenizer


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-100000000):
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    logits = torch.where(logits.isnan(), filter_value, logits)
    logits = torch.where(logits == float('Inf'), filter_value, logits)
    return logits


def get_embedding(model, idx, pos=None):
    token_embed = model.state_dict()['transformer.wte.weight'][idx]
    # pos_embed = model.state_dict()['transformer.wpe.weight'][pos]
    return token_embed


def sample_sequence(hps, model, batch, length, n_ctx, tokenizer: GPT2Tokenizer, temperature=1.0, top_k=30, top_p=0.0, repitition_penalty=1.0):
    context, atten_mask = batch[:2]
    generated = context # B x S x H
    generated_ids = []
    with torch.no_grad():
        for i in range(length):
            inputs = {'inputs_embeds': generated[:, -(n_ctx - 1):], 'attention_mask': atten_mask[:, -(n_ctx - 1):]}
            outputs = model(**inputs)
            if i == 0:
                idx = torch.sum(atten_mask == 1, 1) - 1
                next_token_logits = outputs[0][range(atten_mask.size(0)), idx, :]
            else:
                next_token_logits = outputs[0][:, -1, :]

            next_token_logits = next_token_logits / temperature
            unk = torch.FloatTensor([-float('Inf')]*atten_mask.size(0)).to(atten_mask.device)
            next_token_logits[:, tokenizer.convert_tokens_to_ids('<|endoftext|>')] = unk
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
            # generated = torch.cat((generated, next_token), dim=1)
            generated_ids.append(next_token)
            generated = torch.cat([generated, get_embedding(model, next_token, torch.LongTensor([atten_mask.size(1)]*atten_mask.size(0)).to(atten_mask.device).unsqueeze(1))], 1)
            atten_mask = torch.cat([atten_mask, torch.ones(atten_mask.size(0), 1).to(atten_mask.device)], 1)
    generated_ids = torch.cat(generated_ids, 1)
    return generated_ids


softmax = torch.nn.LogSoftmax(-1)

def sample_sequence_continual(model, batch, length, n_ctx, tokenizer: GPT2Tokenizer, temperature=1.0, top_k=8, top_p=0.0):
    context, atten_mask = batch[:2] # context: B * S * H, atten_mask: B * S
    context = torch.cat([context, context, context], 0)
    atten_mask = torch.cat([atten_mask, atten_mask, atten_mask], 0)
    generated = context # B x S
    generated_ids = []
    dists = []
    index = [-1 for _ in range(context.size(0))]
    # with torch.no_grad():
    for i in range(length):
        # inputs = {'inputs_embeds': generated[:, -(n_ctx - 1):], 'attention_mask': atten_mask[:, -(n_ctx - 1):]}
        inputs = {'inputs_embeds': generated, 'attention_mask': atten_mask}
        outputs = model(**inputs)
        if i == 0:
            idx = torch.sum(atten_mask == 1, 1) - 1
            next_token_logits = outputs[0][range(atten_mask.size(0)), idx, :]
        else:
            next_token_logits = outputs[0][:, -1, :]
        next_token_logits = next_token_logits / temperature
        dists.append(softmax(next_token_logits))
        unk = torch.FloatTensor([-float('Inf')]*atten_mask.size(0)).to(atten_mask.device)
        next_token_logits[:, tokenizer.convert_tokens_to_ids('<|endoftext|>')] = unk
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1) # 3B * 1

        dot_index = (next_token.squeeze() == tokenizer._convert_token_to_id('.')).nonzero().squeeze().cpu().tolist()
        if isinstance(dot_index, int):
            dot_index = [dot_index]
        if len(dot_index) != 0:
            for idx in dot_index:
                index[idx] = i + 1 if index[idx] == -1 else index[idx]

        generated_ids.append(next_token)
        generated = torch.cat([generated, get_embedding(model, next_token)], 1)
        atten_mask = torch.cat([atten_mask, torch.ones(atten_mask.size(0), 1).to(atten_mask.device)], 1)
    
    generated_ids = torch.cat(generated_ids, 1) # 3B * L
    results = []
    batch_size = context.size(0) // 3
    for i in range(batch_size):
        results.append(generated_ids[i::batch_size, :])
    results = torch.stack(results, 0)
    dists = torch.stack(dists, 1)
    return results, dists, index



def sample_sequence_continual2(model, batch, length, n_ctx, tokenizer, temperature=1.0, top_k=8, top_p=0.0):
    context, atten_mask = batch[:2] # context: B * S * H, atten_mask: B * S
    # context = torch.cat([context, context, context], 0)
    # atten_mask = torch.cat([atten_mask, atten_mask, atten_mask], 0)
    generated = context # B x S
    generated_ids = []
    dists = []
    index = [-1 for _ in range(context.size(0)*3)]
    # with torch.no_grad():
    for i in range(length):
        inputs = {'inputs_embeds': generated, 'attention_mask': atten_mask}
        outputs = model(**inputs)
        if i == 0:
            idx = torch.sum(atten_mask == 1, 1) - 1
            next_token_logits = outputs[0][range(atten_mask.size(0)), idx, :]
        else:
            next_token_logits = outputs[0][:, -1, :]
        
        next_token_logits = next_token_logits / temperature # B x V
        
        if i == 0:
            dists.append(softmax(torch.cat([next_token_logits, next_token_logits, next_token_logits], 0)))
        else:
            dists.append(softmax(next_token_logits))
        
        unk = torch.FloatTensor([-float('Inf')]*atten_mask.size(0)).to(atten_mask.device)
        next_token_logits[:, tokenizer.convert_tokens_to_ids('<|endoftext|>')] = unk
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

        # pdb.set_trace()
        if i == 0:
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=3).view(-1, 1) # 3B x 1
            generated = torch.cat([context, context, context], 0) # 3B x S x H
            atten_mask = torch.cat([atten_mask, atten_mask, atten_mask], 0) # 3B x S
        else:
            # repetition penalty
            filtered_logits[range(filtered_logits.size(0)), generated_ids[-1].squeeze(1)] /= 2 # 3B x V

            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1) # 3B * 1

        dot_index = (next_token.squeeze() == tokenizer._convert_token_to_id('.')).nonzero().squeeze().cpu().tolist()
        
        if isinstance(dot_index, int):
            dot_index = [dot_index]
        if len(dot_index) != 0:
            for idx in dot_index:
                index[idx] = i + 1 if index[idx] == -1 else index[idx]

        generated_ids.append(next_token)
        # generated = torch.cat([generated, get_embedding(model, next_token, torch.LongTensor([atten_mask.size(1)]*atten_mask.size(0)).to(atten_mask.device).unsqueeze(1))], 1)
        generated = torch.cat([generated, get_embedding(model, next_token)], 1)
        atten_mask = torch.cat([atten_mask, torch.ones(atten_mask.size(0), 1).to(atten_mask.device)], 1)
    
    generated_ids = torch.cat(generated_ids, 1) # 3B * L
    results = []
    batch_size = context.size(0)
    for i in range(batch_size):
        results.append(generated_ids[i::batch_size, :])
    results = torch.stack(results, 0)
    dists = torch.stack(dists, 1)
    # pdb.set_trace()
    return results, dists, index



def sample_sequence_continual2_2(model, batch, length, n_ctx, tokenizer, temperature=1.0, top_k=8, top_p=0.0):
    context, atten_mask = batch[:2] # context: B * S * H, atten_mask: B * S

    generated = context # B x S
    generated_ids = []
    dists = []
    index = [-1 for _ in range(context.size(0)*3)]

    for i in range(length):
        inputs = {'inputs_embeds': generated, 'attention_mask': atten_mask}
        outputs = model(**inputs)
        if i == 0:
            idx = torch.sum(atten_mask == 1, 1) - 1
            next_token_logits = outputs[0][range(atten_mask.size(0)), idx, :]
        else:
            next_token_logits = outputs[0][:, -1, :]
        
        next_token_logits = next_token_logits / temperature # B x V
        
        if i == 0:
            dists.append(softmax(torch.cat([next_token_logits, next_token_logits], 0)))
        else:
            dists.append(softmax(next_token_logits))
        
        unk = torch.FloatTensor([-float('Inf')]*atten_mask.size(0)).to(atten_mask.device)
        next_token_logits[:, tokenizer.convert_tokens_to_ids('<|endoftext|>')] = unk
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

        if i == 0:
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=2).view(-1, 1) # 2B x 1
            generated = torch.cat([context, context], 0) # 2B x S x H
            atten_mask = torch.cat([atten_mask, atten_mask], 0) # 2B x S
        else:
            # repetition penalty
            filtered_logits[range(filtered_logits.size(0)), generated_ids[-1].squeeze(1)] /= 2 # 2B x V

            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1) # 2B * 1

        dot_index = (next_token.squeeze() == tokenizer._convert_token_to_id('.')).nonzero().squeeze().cpu().tolist()
        
        if isinstance(dot_index, int):
            dot_index = [dot_index]
        if len(dot_index) != 0:
            for idx in dot_index:
                index[idx] = i + 1 if index[idx] == -1 else index[idx]

        generated_ids.append(next_token)
        generated = torch.cat([generated, get_embedding(model, next_token)], 1)
        atten_mask = torch.cat([atten_mask, torch.ones(atten_mask.size(0), 1).to(atten_mask.device)], 1)
    
    generated_ids = torch.cat(generated_ids, 1) # 2B * L
    results = []
    batch_size = context.size(0)
    for i in range(batch_size):
        results.append(generated_ids[i::batch_size, :])
    results = torch.stack(results, 0)
    dists = torch.stack(dists, 1)
    return results, dists, index


def sample_sequence_continual2_1(model, batch, length, n_ctx, tokenizer, temperature=1.0, top_k=8, top_p=0.0):
    context, atten_mask = batch[:2] # context: B * S * H, atten_mask: B * S

    generated = context # B x S
    generated_ids = []
    dists = []
    index = [-1 for _ in range(context.size(0)*3)]

    for i in range(length):
        inputs = {'inputs_embeds': generated, 'attention_mask': atten_mask}
        outputs = model(**inputs)
        if i == 0:
            idx = torch.sum(atten_mask == 1, 1) - 1
            next_token_logits = outputs[0][range(atten_mask.size(0)), idx, :]
        else:
            next_token_logits = outputs[0][:, -1, :]
        
        next_token_logits = next_token_logits / temperature # B x V
        
        dists.append(softmax(next_token_logits))
        
        unk = torch.FloatTensor([-float('Inf')]*atten_mask.size(0)).to(atten_mask.device)
        next_token_logits[:, tokenizer.convert_tokens_to_ids('<|endoftext|>')] = unk
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

        if i == 0:
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1).view(-1, 1) # B x 1
        else:
            # repetition penalty
            filtered_logits[range(filtered_logits.size(0)), generated_ids[-1].squeeze(1)] /= 2 # B x V

            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1) # B * 1

        dot_index = (next_token.squeeze() == tokenizer._convert_token_to_id('.')).nonzero().squeeze().cpu().tolist()
        
        if isinstance(dot_index, int):
            dot_index = [dot_index]
        if len(dot_index) != 0:
            for idx in dot_index:
                index[idx] = i + 1 if index[idx] == -1 else index[idx]

        generated_ids.append(next_token)
        generated = torch.cat([generated, get_embedding(model, next_token)], 1)
        atten_mask = torch.cat([atten_mask, torch.ones(atten_mask.size(0), 1).to(atten_mask.device)], 1)
    
    generated_ids = torch.cat(generated_ids, 1) # 2B * L
    results = []
    batch_size = context.size(0)
    for i in range(batch_size):
        results.append(generated_ids[i::batch_size, :])
    results = torch.stack(results, 0)
    dists = torch.stack(dists, 1)
    return results, dists, index






