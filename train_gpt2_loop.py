import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from argparse import ArgumentParser
import tqdm
from utils.tools import read_data, evaluation_loop, read_retrieved_knowledge
from utils.logger import define_logger
from utils.Dataset import HybridDataset, DynamicDataset
from torch.utils.data import DataLoader, RandomSampler
from module.gpt2_loop import induction, primal, dual
from module.inductive import MultiHeadAttenInduction, HardKumaInduction, PromptInduction, ChainInduction, CometInduction
from torch.optim import AdamW
import os
import random
import numpy as np
import json
import pdb
from torch.utils.tensorboard import SummaryWriter


def hyper_parameters():
    parser = ArgumentParser(description='dual learning gpt2')

    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--comet_dir', type=str, default='')
    parser.add_argument('--log_dir', type=str, default='./logger')
    parser.add_argument('--train', type=str, default='train_gen.jsonl')
    parser.add_argument('--test', type=str, default='test_gen.jsonl')
    parser.add_argument('--dev', type=str, default='dev_gen.jsonl')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--induction', type=str, default='comet', choices=['hidden', 'prompt', 'atten', 'chain', 'comet'])
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
    parser.add_argument('--gpu', type=str, default='[3, 0, 2, 1]')
    parser.add_argument('--log_name', type=str, default='gpt2_loop_comet.txt')
    parser.add_argument('--tb_dir', type=str, default='./tensorboard/gpt2_loop_comet')
    parser.add_argument('--deductive_output', type=str, default='./output/gpt2_loop_comet_deductive.json')
    parser.add_argument('--abductive_output', type=str, default='./output/gpt2_loop_comet_abductive.json')
    parser.add_argument('--lambda1', type=float, default=0.5)
    parser.add_argument('--lambda2', type=float, default=0.5)

    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--repetition_penalty', type=float, default=1.5)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3)
    parser.add_argument('--max_len', type=int, default=15)
    parser.add_argument('--seed', type=int, default=3184)

    opt = parser.parse_args()
    return opt


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
    tokenizer.pad_token = tokenizer.unk_token
    # tokenizer.padding_side = 'left'
    
    abductive = GPT2LMHeadModel.from_pretrained(hps.model_dir)
    deductive = GPT2LMHeadModel.from_pretrained(hps.model_dir)
    
    if hps.induction == 'atten':
        logger.info('[Indution]: atten')
        inductive_de = MultiHeadAttenInduction(hps, abductive.config.n_embd, tokenizer)
        inductive_ab = MultiHeadAttenInduction(hps, abductive.config.n_embd, tokenizer)
    elif hps.induction == 'prompt':
        logger.info('[Indution]: prompt')
        inductive_de = PromptInduction(hps, abductive.config.n_embd, tokenizer)
        inductive_ab = PromptInduction(hps, abductive.config.n_embd, tokenizer)
    elif hps.induction == 'hidden':
        logger.info('[Indution]: hidden')
        inductive_de = HardKumaInduction(hps, abductive.config.n_embd, tokenizer)
        inductive_ab = HardKumaInduction(hps, abductive.config.n_embd, tokenizer)
    elif hps.induction == 'comet':
        logger.info('[Induction]: comet')
        inductive_de = CometInduction(hps, abductive.config.n_embd, tokenizer)
        inductive_ab = CometInduction(hps, abductive.config.n_embd, tokenizer)
    else:
        logger.info('[Indution]: chain')
        inductive_de = ChainInduction(hps, abductive.config.n_embd, tokenizer)
        inductive_ab = ChainInduction(hps, abductive.config.n_embd, tokenizer)
    
    if hps.use_gpu:
        abductive = abductive.cuda(hps.gpu[0])
        deductive = deductive.cuda(hps.gpu[0])
        inductive_de = inductive_de.cuda(hps.gpu[0])
        inductive_ab = inductive_ab.cuda(hps.gpu[0])

    optimizer = AdamW([{'params': abductive.parameters()}, 
                       {'params': deductive.parameters()}, 
                       {'params': inductive_de.parameters()}, 
                       {'params': inductive_ab.parameters()}], lr=hps.lr)

    train_data = read_data(os.path.join(hps.data_dir, hps.train), portion=hps.portion)
    test_data = read_data(os.path.join(hps.data_dir, hps.test))
    dev_data = read_data(os.path.join(hps.data_dir, hps.dev))

    train_kg = read_retrieved_knowledge('./data/train_cpnet.json', portion=hps.portion)
    test_kg = read_retrieved_knowledge('./data/test_cpnet.json')
    dev_kg = read_retrieved_knowledge('./data/dev_cpnet.json')

    # TRAIN = HybridDataset(*train_data)
    # TEST = HybridDataset(*test_data)
    # DEV = HybridDataset(*dev_data)

    train_data = train_data + train_kg
    test_data = test_data + test_kg
    dev_data = dev_data + dev_kg
    TRAIN = DynamicDataset(*train_data)
    TEST = DynamicDataset(*test_data)
    DEV = DynamicDataset(*dev_data)

    sampler = RandomSampler(DEV, replacement=False, num_samples=500)
    train_loader = DataLoader(TRAIN, batch_size=hps.batch_size)
    test_loader = DataLoader(TEST, batch_size=hps.batch_size)
    dev_loader = DataLoader(DEV, batch_size=hps.batch_size, sampler=sampler)

    best_bleu = 0
    patient = 0
    steps = 0
    stop_train = False

    for epoch in range(hps.epochs):
        epoch_step = 0
        bar = tqdm.trange(len(train_loader))
        abductive.train()
        deductive.train()
        total_loss = 0
        
        for batch, _ in zip(train_loader, bar):
            optimizer.zero_grad()
            # pdb.set_trace()
            # primal loop
            loss1, loss2 = primal(hps, abductive, deductive, inductive_de, inductive_de, batch, tokenizer)
            if hps.induction == 'hidden':
                loss3 = inductive_de.get_loss() + inductive_ab.get_loss()
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
            optimizer.zero_grad()
            loss1, loss2 = dual(hps, abductive, deductive, inductive_de, inductive_de, batch, tokenizer)
            if hps.induction == 'hidden':
                loss3 = inductive_de.get_loss() + inductive_ab.get_loss()
            else:
                loss3 = 0
            loss_dual = hps.lambda1 * loss1 + hps.lambda2 * loss2 + loss3
            if loss_dual.isnan().item():
                continue
            loss_dual.backward()
            optimizer.step()
            total_loss += loss_primal.item()

            if steps % 20 == 0:
                writer.add_scalar('abductive_loss', loss1.detach(), steps // 20)

            epoch_step += 1
            steps += 1

            bar.set_postfix(loss='{}'.format((loss_primal.item()+loss_dual.item())/2))
            # dev_ab = evaluation_loop(hps, abductive, inductive, dev_loader, tokenizer, mode='abductive')

            # if steps != 0 and epoch_step != 0 and steps % hps.evaluation_steps == 0 or epoch_step % len(bar) == 0:
        logger.info('[Evaluation] Starting evaluation on dev set')

        dev_ab = evaluation_loop(hps, abductive, inductive_de, dev_loader, tokenizer, mode='abductive')
        logger.info("[Dev Abduction Rouge-l]: {}".format(dev_ab[1]))
        logger.info('[Dev Abduction Average Bleu]: {}'.format(dev_ab[0]))

        writer.add_scalar('Dev abductive Rouge-l', dev_ab[1], steps // hps.evaluation_steps)
        writer.add_scalar('Dev abductive Average Bleu', dev_ab[0], steps // hps.evaluation_steps)

        dev_de = evaluation_loop(hps, deductive, inductive_de, dev_loader, tokenizer, mode='deductive')
        logger.info("[Dev Deduction Rouge-l]: {}".format(dev_de[1]))
        logger.info('[Dev Deduction Average Bleu]: {}'.format(dev_de[0]))

        writer.add_scalar('Dev deductive Rouge-l', dev_de[1], steps // hps.evaluation_steps)
        writer.add_scalar('Dev deductive Average Bleu', dev_de[0], steps // hps.evaluation_steps)
        
        if dev_ab[0] + dev_de[0] + (dev_ab[1] + dev_de[1]) / 2 >= best_bleu:
            best_bleu = dev_ab[0] + dev_de[0] + (dev_ab[1] + dev_de[1]) / 2
            patient = 0
            logger.info('[Evaluation] Starting evaluation on test set')

            output_de_fi = open(hps.deductive_output, 'w')
            output_ab_fi = open(hps.abductive_output, 'w')

            test_ab = evaluation_loop(hps, abductive, inductive_de, test_loader, tokenizer, mode='abductive')
            logger.info("[Test Abductive Rouge-l]: {}".format(test_ab[1]))
            logger.info('[Test Abductive Average Bleu]: {}'.format(test_ab[0]))

            json.dump(test_ab[-1], output_ab_fi, indent=1)

            test_de = evaluation_loop(hps, deductive, inductive_de, test_loader, tokenizer, mode='deductive')
            logger.info("[Test Deductive Rouge-l]: {}".format(test_de[1]))
            logger.info('[Test Deductive Average Bleu]: {}'.format(test_de[0]))
            torch.save({'deductive_dict': deductive.state_dict(),
                        'abductive_dict': abductive.state_dict(),
                        'indective_de_dict': inductive_de.state_dict(),
                        'indective_ab_dict': inductive_ab.state_dict(),
                        'optimizer_dict': optimizer.state_dict()}, 
                        './output/model/gpt2_loop_{}_portion{}.ckpt'.format(hps.induction, round(hps.portion*100)))
            json.dump(test_de[-1], output_de_fi, indent=1)
        
        else:
            patient += 1
            if patient == hps.patient:
                stop_train = True
                break
        
        if stop_train:
            break




























