import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from argparse import ArgumentParser
import tqdm
from utils.tools import tokenize_gpt2
from utils.logger import define_logger
from torch.optim import AdamW
import os
import random
import numpy as np
from utils.Dataset import DynamicDataset
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from rouge import Rouge
from nltk import bleu
from nltk.tokenize import word_tokenize
import jsonlines




def hyper_parameters():
    parser = ArgumentParser(description='dual learning gpt2')

    parser.add_argument('--data_dir', type=str, default='./data/anlg')
    parser.add_argument('--model_dir', type=str, default='/ssd1/huggingface_transformers/gpt2')
    parser.add_argument('--log_dir', type=str, default='./logger')
    parser.add_argument('--train', type=str, default='train-w-comet-preds.jsonl')
    parser.add_argument('--dev', type=str, default='dev-w-comet-preds.jsonl')
    parser.add_argument('--test', type=str, default='test-w-comet-preds.jsonl')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--evaluation_steps', type=int, default=200)
    parser.add_argument('--mode', type=str, default='deductive')
    parser.add_argument('--patient', type=int, default=10)
    parser.add_argument('--portion', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=str, default='[1, 0, 2, 3]')
    parser.add_argument('--log_name', type=str, default='gpt2_aNLG_deductive.txt')
    parser.add_argument('--lambda1', type=float, default=0.99)
    parser.add_argument('--lambda2', type=float, default=0.01)

    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--repetition_penalty', type=float, default=1.5)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3)
    parser.add_argument('--max_len', type=int, default=30)
    parser.add_argument('--seed', type=int, default=3184)

    opt = parser.parse_args()
    return opt


def read_jsonl(path):
    data = jsonlines.open(path, 'r')
    O1, H, O2 = [], [], []
    for example in data:
        O1.append(example['obs1'])
        O2.append(example['obs2'])
        H.append(example['hyp1'] if example['label'] == '1' else example['hyp2'])
    return O1, H, O2


def evaluation(hps, model: GPT2LMHeadModel, dataloader, tokenizer: GPT2Tokenizer):
    bleu_score, rougel = 0, 0
    bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
    num_instances = 0
    predictions = []
    rouge = Rouge()
    model.eval()
    bar = tqdm.trange(len(dataloader))

    for _, batch in zip(bar, dataloader):
        O1, H, O2 = batch
        if hps.mode == 'deductive':
            inputs = ["{} {}".format(o1, h) for o1, h in zip(O1, H)]
            outputs = O2
        else:
            inputs = ["{} {}".format(o1, o2) for o1, o2 in zip(O1, O2)]
            outputs = H
        
        num_instances += len(inputs)

        inputs_t = tokenizer(inputs, padding=True, return_tensors='pt')

        inputs_t = [inputs_t[term].cuda(hps.gpu[0]) for term in inputs_t]

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids = inputs_t[0],
                attention_mask = inputs_t[1],
                do_sample=True,
                repetition_penalty=2.0,
                pad_token_id=tokenizer.eos_token_id,
                max_length=inputs_t[0].size(1)+hps.max_len
            )

        generated_text = tokenizer.batch_decode(generated_ids[:, inputs_t[0].size(1):], clean_up_tokenization_spaces=True,skip_special_tokens=True)
        generated_text = [sent.split('.')[0]+'.' for sent in generated_text]

        for pre, hypo, ref in zip(inputs, generated_text, outputs):
            if hps.mode == 'deductive':
                predictions.append({'cause': pre, 'effect': ref, 'prediction': hypo})
            else:
                predictions.append({'cause': ref, 'effect': pre, 'prediction': hypo})
            
            try:
                rougel += rouge.get_scores(hypo, ref)[0]['rouge-l']['f']
            except:
                continue
            
            ref = word_tokenize(ref)
            hypo = word_tokenize(hypo)

            bleu_score += bleu([ref], hypo, weights=[0.25, 0.25, 0.25, 0.25])
            bleu1 += bleu([ref], hypo, weights=[1, 0, 0, 0])
            bleu2 += bleu([ref], hypo, weights=[0, 1, 0, 0])
            bleu3 += bleu([ref], hypo, weights=[0, 0, 1, 0])
            bleu4 += bleu([ref], hypo, weights=[0, 0, 0, 1])

    bleu1 /= num_instances
    bleu2 /= num_instances
    bleu3 /= num_instances
    bleu4 /= num_instances

    return bleu_score / num_instances, rougel / num_instances, (bleu1, bleu2, bleu3, bleu4), predictions


def tokenization(hps, tokenizer: GPT2Tokenizer, O1, H, O2):
    if hps.mode == 'deductive':
        inputs = ["{} {}".format(o1, h) for o1, h in zip(O1, H)]
        outputs = O2
    else:
        inputs = ["{} {}".format(o1, o2) for o1, o2 in zip(O1, O2)]
        outputs = H

    return tokenize_gpt2(inputs, outputs, tokenizer)


if __name__ == '__main__':    
    hps = hyper_parameters()
    hps.gpu = eval(hps.gpu)
    logger = define_logger(hps)

    logger.info(hps)

    torch.manual_seed(hps.seed)
    torch.cuda.manual_seed(hps.seed)
    random.seed(hps.seed)
    np.random.seed(hps.seed)

    logger.info('[Mode] Using {} as backend.'.format(hps.model_name))
    logger.info('[GPU] Using {} for training.'.format(hps.gpu))
    logger.info('[Init] Initializing Models, Optimizer & Tokenizer')
    logger.info('[Mode] {}'.format(hps.mode))

    model = GPT2LMHeadModel.from_pretrained(hps.model_dir)

    model = model.cuda(hps.gpu[0])

    optimizer = AdamW(model.parameters(), lr=hps.lr)
    tokenizer = GPT2Tokenizer.from_pretrained(hps.model_dir)
    tokenizer.pad_token = tokenizer.unk_token

    train_data = read_jsonl(os.path.join(hps.data_dir, hps.train))
    dev_data = read_jsonl(os.path.join(hps.data_dir, hps.dev))
    test_data = read_jsonl(os.path.join(hps.data_dir, hps.test))

    TRAIN = TensorDataset(*tokenization(hps, tokenizer, *train_data))
    DEV = DynamicDataset(*dev_data)
    TEST = DynamicDataset(*test_data)
    
    train_loader = DataLoader(TRAIN, batch_size=hps.batch_size)
    dev_loader = DataLoader(DEV, batch_size=hps.batch_size)
    test_loader = DataLoader(TEST, batch_size=hps.batch_size)

    best_score = 0
    patient = 0
    steps = 0
    stop_train = False

    for epoch in range(hps.epochs):
        epoch_step = 0
        bar = tqdm.trange(len(train_loader))
        model.train()

        total_loss = 0
        
        for batch, _ in zip(train_loader, bar):
            optimizer.zero_grad()

            batch = [term.cuda(hps.gpu[0]) for term in batch]

            loss = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2]).loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            epoch_step += 1
            steps += 1

            bar.set_postfix(loss='{}'.format(total_loss/epoch_step))

        logger.info('[Evaluation] Starting evaluation on dev set')

        dev_scores = evaluation(hps, model, dev_loader, tokenizer)
        logger.info("[Dev {} Rouge-l]: {}".format(hps.mode, dev_scores[1]))
        logger.info('[Dev {} Bleu1]: {}'.format(hps.mode, dev_scores[2][0]))
        logger.info('[Dev {} Bleu2]: {}'.format(hps.mode, dev_scores[2][1]))
        logger.info('[Dev {} Bleu3]: {}'.format(hps.mode, dev_scores[2][2]))
        logger.info('[Dev {} Bleu4]: {}'.format(hps.mode, dev_scores[2][3]))
        logger.info('[Dev {} Average Bleu]: {}'.format(hps.mode, dev_scores[0]))

        logger.info('[Evaluation] Starting evaluation on test set')
        test_scores = evaluation(hps, model, test_loader, tokenizer)
        logger.info("[Test {} Rouge-l]: {}".format(hps.mode, dev_scores[1]))
        logger.info('[Test {} Bleu1]: {}'.format(hps.mode, dev_scores[2][0]))
        logger.info('[Test {} Bleu2]: {}'.format(hps.mode, dev_scores[2][1]))
        logger.info('[Test {} Bleu3]: {}'.format(hps.mode, dev_scores[2][2]))
        logger.info('[Test {} Bleu4]: {}'.format(hps.mode, dev_scores[2][3]))
        logger.info('[Test {} Average Bleu]: {}'.format(hps.mode, dev_scores[0]))

        if test_scores[1] >= best_score:
            output_fi = open('./output/{}_{}.json'.format(hps.model_name, hps.mode), 'w')
            best_score = test_scores[1]
            patient = 0

            torch.save({'model_dict': model.state_dict()},
                        './output/model/{}_{}_{}.ckpt'.format(hps.model_name, hps.mode, 'aNLG'))

            json.dump(test_scores[-1], output_fi, indent=1)
            logger.info('[Patient]: {}'.format(patient))
        
        else:
            patient += 1
            logger.info('[Patient]: {}'.format(patient))
            continue
            # patient += 1
            # if patient == hps.patient:
            #     stop_train = True
            #     break
        
        # if stop_train:
        #     break
    




























