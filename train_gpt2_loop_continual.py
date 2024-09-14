import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from argparse import ArgumentParser
import tqdm
from utils.tools import read_data, evaluation_loop, read_retrieved_knowledge
from utils.logger import define_logger
from utils.Dataset import DynamicDataset
from torch.utils.data import DataLoader, RandomSampler
from module.gpt2_loop_continual import primal, dual
from module.gpt2_loop import primal as primal_pa
from module.gpt2_loop import dual as dual_pa
from module.inductive import MultiHeadAttenInduction, HardKumaInduction, PromptInduction, ChainInduction, CometInduction
from torch.optim import AdamW
import os
import random
import numpy as np
import json
import pdb
from nltk.tokenize import word_tokenize
from torch.utils.tensorboard import SummaryWriter


def hyper_parameters():
    parser = ArgumentParser(description='dual learning gpt2')

    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='.')
    parser.add_argument('--causality', type=str, default='')
    parser.add_argument('--comet_dir', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='./logger')
    parser.add_argument('--train', type=str, default='train_gen.jsonl')
    parser.add_argument('--test', type=str, default='test_gen.jsonl')
    parser.add_argument('--dev', type=str, default='dev_gen.jsonl')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--induction', type=str, default='atten', choices=['hidden', 'prompt', 'atten', 'chain', 'comet'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--evaluation_steps', type=int, default=200)
    parser.add_argument('--patient', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--portion', type=float, default=1.0)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--e2e', type=bool, default=False)
    parser.add_argument('--multi_sent', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--gpu', type=str, default='[0, 3, 2, 1]')
    parser.add_argument('--log_name', type=str, default='gpt2_loop_atten_continual.txt')
    parser.add_argument('--tb_dir', type=str, default='./tensorboard/gpt2_loop_atten_continual')
    parser.add_argument('--deductive_output', type=str, default='./output/gpt2_loop_atten_continual_deductive.json')
    parser.add_argument('--abductive_output', type=str, default='./output/gpt2_loop_atten_continual_abductive.json')
    parser.add_argument('--lambda1', type=float, default=0.5)
    parser.add_argument('--lambda2', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.99)

    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--repetition_penalty', type=float, default=1.5)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3)
    parser.add_argument('--max_len', type=int, default=15)
    parser.add_argument('--seed', type=int, default=3184)

    opt = parser.parse_args()
    return opt


def read_augument_data(path):
    data = json.load(open(path, 'r'))
    sents, void, entities, chains = [], [], [], []
    data = sorted(data, key=lambda d: len(word_tokenize(d['sent'])))
    # pdb.set_trace()
    for instance in data:
        words = word_tokenize(instance['sent'])
        if len(words) <= 3 or len(words) > 15:
            continue 
        if 'it' in words:
            continue
        sents.append(instance['sent'] + '.' if instance['sent'][-1] not in ['.', '?', '!'] else instance['sent'])
        entities.append(' '.join(instance['entities']))
        chains.append(';'.join(instance['chains'][0]))
        void.append('None')
    return sents, void, entities, chains


if __name__ == '__main__':    
    hps = hyper_parameters()
    hps.gpu = eval(hps.gpu)
    logger = define_logger(hps)

    logger.info(hps)
    writer = SummaryWriter(log_dir=hps.tb_dir)

    torch.manual_seed(hps.seed)
    torch.cuda.manual_seed(hps.seed)
    random.seed(hps.seed)
    np.random.seed(hps.seed)

    logger.info('[Mode] Using {} as backend.'.format(hps.model_name))
    logger.info('[GPU] Using {} for training.'.format(hps.gpu))
    logger.info('[Init] Initializing Models, Optimizer & Tokenizer')

    tokenizer = GPT2Tokenizer.from_pretrained(hps.model_dir)
    config = GPT2Config.from_pretrained(hps.model_dir)
    tokenizer.pad_token = tokenizer.unk_token
    
    abductive = GPT2LMHeadModel(config)
    deductive = GPT2LMHeadModel(config)

    causality = RobertaForSequenceClassification.from_pretrained(hps.causality)
    tokenizer_c = RobertaTokenizer.from_pretrained(hps.causality)

    
    if hps.induction == 'atten':
        logger.info('[Indution]: atten')
        inductive = MultiHeadAttenInduction(hps, abductive.config.n_embd, tokenizer)
    elif hps.induction == 'prompt':
        logger.info('[Indution]: prompt')
        inductive = PromptInduction(hps, abductive.config.n_embd, tokenizer)
    elif hps.induction == 'hidden':
        logger.info('[Indution]: hidden')
        inductive = HardKumaInduction(hps, abductive.config.n_embd, tokenizer)
    elif hps.induction == 'comet':
        logger.info('[Induction]: comet')
        inductive = CometInduction(hps, abductive.config.n_embd, tokenizer)
    else:
        logger.info('[Indution]: chain')
        inductive = ChainInduction(hps, abductive.config.n_embd, tokenizer)
    
    checkpoint = torch.load(hps.checkpoint, map_location='cpu')
    abductive.load_state_dict(checkpoint['abductive_dict'])
    deductive.load_state_dict(checkpoint['deductive_dict'])
    inductive.load_state_dict(checkpoint['indective_dict'])


    abductive = abductive.cuda(hps.gpu[0])
    deductive = deductive.cuda(hps.gpu[0])
    inductive = inductive.cuda(hps.gpu[0])
    causality = causality.cuda(hps.gpu[0])

    optimizer = AdamW([{'params': abductive.parameters()}, {'params': deductive.parameters()}, {'params': inductive.parameters()}], lr=hps.lr)
    optimizer2 = AdamW([{'params': abductive.parameters()}, {'params': deductive.parameters()}, {'params': inductive.parameters()}], lr=hps.lr)

    test_data = read_data(os.path.join(hps.data_dir, hps.test))
    parallel_data = read_data(os.path.join(hps.data_dir, hps.train))

    test_kg = read_retrieved_knowledge('./data/test_cpnet.json')
    parallel_kg = read_retrieved_knowledge('./data/train_cpnet.json')

    train_data = read_augument_data('./data/story_cpnet.json')
    test_data = test_data + test_kg
    parallel_data = parallel_data + parallel_kg
    TRAIN = DynamicDataset(*train_data)
    TEST = DynamicDataset(*test_data)
    PARALLEL = DynamicDataset(*parallel_data)

    parallel_sampler = RandomSampler(PARALLEL, replacement=True, num_samples=1000)
    train_loader = DataLoader(TRAIN, batch_size=hps.batch_size)
    test_loader = DataLoader(TEST, batch_size=hps.batch_size)
    parallel_loader = DataLoader(PARALLEL, batch_size=hps.batch_size, sampler=parallel_sampler)

    best_bleu = 0
    patient = 0
    steps = 0
    stop_train = False
    primal_fo = open('{}_primal.json'.format(hps.log_name[9:-4]), 'w')
    dual_fo = open('{}_dual.json'.format(hps.log_name[9:-4]), 'w')

    for epoch in range(hps.epochs):
        epoch_step = 0
        bar = tqdm.trange(len(train_loader))
        abductive.train()
        deductive.train()
        inductive.train()
        total_loss = 0
        
        for batch, _ in zip(train_loader, bar):
            optimizer.zero_grad()
            # primal loop
            if steps % 2 == 0: 
                loss1, loss2 = primal(hps, abductive, deductive, inductive, causality, batch, tokenizer, tokenizer_c, primal_fo)
                if hps.induction == 'hidden':
                    loss3 = inductive.get_loss()
                else:
                    loss3 = 0
                loss_primal = hps.lambda1 * loss1 + hps.lambda2 * loss2 + loss3
                if loss_primal.isnan().item():
                    continue
                loss_primal.backward()
                optimizer.step()

                if steps % 20 == 0:
                    writer.add_scalar('deductive_loss', loss1.detach(), steps // 20)

            # dual loop
            else:
                optimizer.zero_grad()
                loss1, loss2 = dual(hps, abductive, deductive, inductive, causality, batch, tokenizer, tokenizer_c, dual_fo)
                if hps.induction == 'hidden':
                    loss3 = inductive.get_loss()
                else:
                    loss3 = 0
                loss_dual = hps.lambda1 * loss1 + hps.lambda2 * loss2 + loss3
                loss_dual.backward()
                if loss_dual.isnan().item():
                    continue
                optimizer.step()
                total_loss += loss_primal.item()

                if steps % 20 == 0:
                    writer.add_scalar('abductive_loss', loss1.detach(), steps // 20)

            epoch_step += 1
            steps += 1

            bar.set_postfix(loss_step='{}_{}'.format(total_loss/epoch_step, steps))

            if steps % 400 == 0:
                logger.info('[Parallel] 1000 Parallel data training')
                for _, batch in zip(tqdm.trange(len(parallel_loader)), parallel_loader):
                    optimizer2.zero_grad()
                    loss1, loss2 = primal_pa(hps, abductive, deductive, inductive, batch, tokenizer)
                    if hps.induction == 'hidden':
                        loss3 = inductive.get_loss()
                    else:
                        loss3 = 0
                    loss_primal = hps.lambda1 * loss1 + hps.lambda2 * loss2 + loss3
                    loss_primal.backward()
                    optimizer2.step()

                    # dual loop
                    optimizer2.zero_grad()
                    loss1, loss2 = dual_pa(hps, abductive, deductive, inductive, batch, tokenizer)
                    if hps.induction == 'hidden':
                        loss3 = inductive.get_loss()
                    else:
                        loss3 = 0
                    loss_dual = hps.lambda1 * loss1 + hps.lambda2 * loss2 + loss3
                    loss_dual.backward()
                    optimizer2.step()
    
                logger.info('[Evaluation] Starting evaluation on test set')

                output_de_fo = open("{}_{}.json".format(hps.deductive_output, steps), 'w')
                output_ab_fo = open("{}_{}.json".format(hps.abductive_output, steps), 'w')

                test_ab = evaluation_loop(hps, abductive, inductive, test_loader, tokenizer, mode='abductive')
                logger.info("[Test Abductive Rouge-l]: {}".format(test_ab[1]))
                logger.info('[Test Abductive Average Bleu]: {}'.format(test_ab[0]))

                json.dump(test_ab[-1], output_ab_fo, indent=1)

                test_de = evaluation_loop(hps, deductive, inductive, test_loader, tokenizer, mode='deductive')
                logger.info("[Test Deductive Rouge-l]: {}".format(test_de[1]))
                logger.info('[Test Deductive Average Bleu]: {}'.format(test_de[0]))
                # torch.save({'deductive_dict': deductive.state_dict(),
                #             'abductive_dict': abductive.state_dict(),
                #             'inductive_dict': inductive.state_dict(),
                #             'optimizer_dict': optimizer.state_dict()}, 
                #             './output/model/gpt2_loop_{}_continual_{}.ckpt'.format(hps.induction, steps))
                json.dump(test_de[-1], output_de_fo, indent=1)
                output_ab_fo.close()
                output_de_fo.close()




























