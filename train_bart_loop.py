import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from argparse import ArgumentParser
import tqdm
from utils.tools import read_data, evaluation_loop_bart, read_retrieved_knowledge
from utils.logger import define_logger
from utils.Dataset import DynamicDataset
from torch.utils.data import DataLoader, RandomSampler
from module.bart_loop import primal, dual
from module.inductive_encoder_decoder import MultiHeadAttenInduction, HardKumaInduction, ChainInduction
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
    parser.add_argument('--log_dir', type=str, default='./logger')
    parser.add_argument('--train', type=str, default='train_gen.jsonl')
    parser.add_argument('--test', type=str, default='test_gen.jsonl')
    parser.add_argument('--dev', type=str, default='dev_gen.jsonl')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--model_name', type=str, default='bart')
    parser.add_argument('--induction', type=str, default='atten', choices=['hidden', 'atten', 'chain'])
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
    parser.add_argument('--log_name', type=str, default='bart_loop_comet.txt')
    parser.add_argument('--tb_dir', type=str, default='./tensorboard/bart_loop_atten')
    parser.add_argument('--save_dir', type=str, default='./output/model/bart_loop_atten.ckpt')
    parser.add_argument('--deductive_output', type=str, default='./output/bart_loop_atten_deductive.json')
    parser.add_argument('--abductive_output', type=str, default='./output/bart_loop_atten_abductive.json')
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

    tokenizer = BartTokenizer.from_pretrained(hps.model_dir)
    
    abductive = BartForConditionalGeneration.from_pretrained(hps.model_dir)
    deductive = BartForConditionalGeneration.from_pretrained(hps.model_dir)
    # abductive.load_state_dict(torch.load('./output/model/bart_abductive.ckpt', map_location='cpu')['model_dict'])
    # deductive.load_state_dict(torch.load('./output/model/bart_deductive.ckpt', map_location='cpu')['model_dict'])
    
    if hps.induction == 'atten':
        logger.info('[Indution]: atten')
        inductive_de = MultiHeadAttenInduction(hps, abductive.config.d_model, tokenizer)
        inductive_ab = MultiHeadAttenInduction(hps, abductive.config.d_model, tokenizer)
    elif hps.induction == 'hidden':
        logger.info('[Indution]: hidden')
        inductive = HardKumaInduction(hps, abductive.config.d_model, tokenizer)
    else:
        logger.info('[Indution]: chain')
        inductive = ChainInduction(hps, abductive.config.d_model, tokenizer)
    
    abductive = abductive.cuda(hps.gpu[0])
    deductive = deductive.cuda(hps.gpu[0])
    inductive_de = inductive_de.cuda(hps.gpu[0])
    inductive_ab = inductive_ab.cuda(hps.gpu[0])

    optimizer = AdamW([{'params': abductive.parameters(), 'lr': hps.lr}, 
                       {'params': deductive.parameters(), 'lr': hps.lr}, 
                       {'params': inductive_de.parameters(), 'lr': hps.lr}, 
                       {'params': inductive_ab.parameters(), 'lr': hps.lr}])

    train_data = read_data(os.path.join(hps.data_dir, hps.train), portion=hps.portion)
    test_data = read_data(os.path.join(hps.data_dir, hps.test))
    dev_data = read_data(os.path.join(hps.data_dir, hps.dev))

    train_kg = read_retrieved_knowledge('./data/train_cpnet.json', portion=hps.portion)
    test_kg = read_retrieved_knowledge('./data/test_cpnet.json')
    dev_kg = read_retrieved_knowledge('./data/dev_cpnet.json')

    train_data = train_data + train_kg
    test_data = test_data + test_kg
    dev_data = dev_data + dev_kg
    TRAIN = DynamicDataset(*train_data)
    TEST = DynamicDataset(*test_data)
    DEV = DynamicDataset(*dev_data)

    # sampler = RandomSampler(DEV, replacement=False, num_samples=500)
    train_loader = DataLoader(TRAIN, batch_size=hps.batch_size)
    test_loader = DataLoader(TEST, batch_size=hps.batch_size)
    # dev_loader = DataLoader(DEV, batch_size=hps.batch_size, sampler=sampler)
    dev_loader = DataLoader(DEV, batch_size=hps.batch_size)

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

            # primal loop
            loss1, loss2 = primal(hps, abductive, deductive, inductive_de, inductive_ab, batch, tokenizer)
            
            loss3 = inductive.get_loss() if hps.induction == 'hidden' else 0

            loss_primal = hps.lambda1 * loss1 + hps.lambda2 * loss2 + loss3
            if loss_primal.isnan().item():
                continue
            loss_primal.backward()
            optimizer.step()

            if steps % 20 == 0:
                writer.add_scalar('deductive_loss', loss1.detach(), steps // 20)

            # dual loop
            optimizer.zero_grad()
            loss1, loss2 = dual(hps, abductive, deductive, inductive_de, inductive_ab, batch, tokenizer)

            loss3 = inductive.get_loss() if hps.induction == 'hidden' else 0

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

        logger.info('[Evaluation] Starting evaluation on dev set')

        dev_ab = evaluation_loop_bart(hps, abductive, inductive_ab, dev_loader, tokenizer, mode='abductive')
        logger.info("[Dev Abduction Rouge-l]: {}".format(dev_ab[1]))
        logger.info('[Dev Abduction Average Bleu]: {}'.format(dev_ab[0]))

        writer.add_scalar('Dev abductive Rouge-l', dev_ab[1], steps // hps.evaluation_steps)
        writer.add_scalar('Dev abductive Average Bleu', dev_ab[0], steps // hps.evaluation_steps)

        dev_de = evaluation_loop_bart(hps, deductive, inductive_de, dev_loader, tokenizer, mode='deductive')
        logger.info("[Dev Deduction Rouge-l]: {}".format(dev_de[1]))
        logger.info('[Dev Deduction Average Bleu]: {}'.format(dev_de[0]))

        writer.add_scalar('Dev deductive Rouge-l', dev_de[1], steps // hps.evaluation_steps)
        writer.add_scalar('Dev deductive Average Bleu', dev_de[0], steps // hps.evaluation_steps)
        
        if dev_ab[1] + dev_de[1] >= best_bleu:
            best_bleu = dev_ab[1] + dev_de[1]
            patient = 0
            logger.info('[Evaluation] Starting evaluation on test set')

            output_de_fi = open(hps.deductive_output, 'w')
            output_ab_fi = open(hps.abductive_output, 'w')

            test_ab = evaluation_loop_bart(hps, abductive, inductive_ab, test_loader, tokenizer, mode='abductive')
            logger.info("[Test Abductive Rouge-l]: {}".format(test_ab[1]))
            logger.info('[Test Abductive Average Bleu]: {}'.format(test_ab[0]))

            json.dump(test_ab[-1], output_ab_fi, indent=1)

            test_de = evaluation_loop_bart(hps, deductive, inductive_de, test_loader, tokenizer, mode='deductive')
            logger.info("[Test Deductive Rouge-l]: {}".format(test_de[1]))
            logger.info('[Test Deductive Average Bleu]: {}'.format(test_de[0]))
            torch.save({'deductive_dict': deductive.state_dict(),
                        'abductive_dict': abductive.state_dict(),
                        'inductive_de_dict': inductive_de.state_dict(),
                        'inductive_ab_dict': inductive_ab.state_dict()}, 
                        hps.save_dir)
            json.dump(test_de[-1], output_de_fi, indent=1)
        
        else:
            patient += 1
            if patient == hps.patient:
                stop_train = True
                break
        
        if stop_train:
            break




























