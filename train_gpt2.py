from asyncore import write
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from argparse import ArgumentParser
import tqdm
from utils.tools import read_data, evaluation_gpt2, tokenize_gpt2, tokenize4gen
from utils.logger import define_logger
from torch.optim import AdamW
import os
import random
import numpy as np
from utils.Dataset import TrainDataset, HybridDataset
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter




def hyper_parameters():
    parser = ArgumentParser(description='dual learning gpt2')

    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--model_dir', type=str, default='/ssd1/huggingface_transformers/gpt2')
    parser.add_argument('--log_dir', type=str, default='./logger')
    parser.add_argument('--train', type=str, default='train_gen.jsonl')
    parser.add_argument('--test', type=str, default='test_gen.jsonl')
    parser.add_argument('--dev', type=str, default='dev_gen.jsonl')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--model_name', type=str, default='gpt2')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--evaluation_steps', type=int, default=200)
    parser.add_argument('--mode', type=str, default='abductive')
    parser.add_argument('--patient', type=int, default=10)
    parser.add_argument('--portion', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=str, default='[0, 1, 2, 3]')
    parser.add_argument('--log_name', type=str, default='dual_learning_gpt2.txt')
    parser.add_argument('--lambda1', type=float, default=0.99)
    parser.add_argument('--lambda2', type=float, default=0.01)

    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--repetition_penalty', type=float, default=1.5)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3)
    parser.add_argument('--max_len', type=int, default=30)
    parser.add_argument('--seed', type=int, default=3184)

    opt = parser.parse_args()
    return opt


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

    model = GPT2LMHeadModel.from_pretrained(hps.model_dir)
    writer = SummaryWriter(log_dir='./tensorboard/gpt2_{}_portion{}'.format(hps.mode, round(hps.portion*100)))

    if hps.use_gpu:
        model = model.cuda(hps.gpu[0])

    optimizer = AdamW(model.parameters(), lr=hps.lr)
    tokenizer = GPT2Tokenizer.from_pretrained(hps.model_dir)
    tokenizer.pad_token = tokenizer.unk_token

    train_data = read_data(os.path.join(hps.data_dir, hps.train), portion=hps.portion)
    test_data = read_data(os.path.join(hps.data_dir, hps.test))
    dev_data = read_data(os.path.join(hps.data_dir, hps.dev))

    if hps.mode == 'deductive':
        train_ids, train_mask, train_labels = tokenize_gpt2(*train_data, tokenizer)
        test_ids, test_mask = tokenize4gen(test_data[0], tokenizer)
        dev_ids, dev_mask = tokenize4gen(dev_data[0], tokenizer)
        test_references, dev_references = test_data[1], dev_data[1]
    else:
        train_ids, train_mask, train_labels = tokenize_gpt2(train_data[1], train_data[0], tokenizer)
        test_ids, test_mask = tokenize4gen(test_data[1], tokenizer)
        dev_ids, dev_mask = tokenize4gen(dev_data[1], tokenizer)
        test_references, dev_references = test_data[0], dev_data[0]

    TRAIN = TrainDataset(train_ids, train_mask, train_labels)
    TEST = TrainDataset(test_ids, test_mask, test_references)
    DEV = TrainDataset(dev_ids, dev_mask, dev_references)

    dev_sampler = RandomSampler(DEV, replacement=False, num_samples=500)
    # test_sampler = RandomSampler(TEST, replacement=False, num_samples=1000)
    train_loader = DataLoader(TRAIN, batch_size=hps.batch_size)
    test_loader = DataLoader(TEST, batch_size=hps.batch_size)
    dev_loader = DataLoader(DEV, batch_size=hps.batch_size, sampler=dev_sampler)

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

            if hps.use_gpu:
                batch = [term.cuda(hps.gpu[0]) for term in batch]

            input_ids, attention_mask, labels = batch

            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            loss.backward()
            optimizer.step()

            if steps % 20 == 0:
                writer.add_scalar("{}_loss".format(hps.mode), loss.detach(), steps//20)

            total_loss += loss.item()
            epoch_step += 1
            steps += 1

            bar.set_postfix(loss='{}'.format(total_loss/epoch_step))
            # dev_scores = evaluation_gpt2(hps, model, dev_loader, tokenizer)

            # if steps != 0 and epoch_step != 0 and steps % hps.evaluation_steps == 0 or epoch_step % len(bar) == 0:
        logger.info('[Evaluation] Starting evaluation on dev set')

        dev_scores = evaluation_gpt2(hps, model, dev_loader, tokenizer)
        logger.info("[Dev {} Rouge-l]: {}".format(hps.mode, dev_scores[1]))
        logger.info('[Dev {} Average Bleu]: {}'.format(hps.mode, dev_scores[0]))
        
        writer.add_scalar("Dev {} Rouge-l".format(hps.mode), dev_scores[1], steps)
        writer.add_scalar("Dev {} Average Bleu".format(hps.mode), dev_scores[0], steps)

        if dev_scores[1] + dev_scores[0] >= best_score:
            output_fi = open('./output/gpt2_{}.json'.format(hps.mode), 'w')
            best_score = dev_scores[1] + dev_scores[0]
            patient = 0
            logger.info('[Evaluation] Starting evaluation on test set')

            test_ab = evaluation_gpt2(hps, model, test_loader, tokenizer)
            logger.info("[Test {} Rouge-l]: {}".format(hps.mode, test_ab[1]))
            logger.info('[Test {} Average Bleu]: {}'.format(hps.mode, test_ab[0]))

            torch.save({'model_dict': model.state_dict(),
                        'optimizer_dict': optimizer.state_dict()},
                        './output/model/gpt2_{}_portion{}.ckpt'.format(hps.mode, hps.portion))

            json.dump(test_ab[-1], output_fi, indent=1)
        
        else:
            patient += 1
            if patient == hps.patient:
                stop_train = True
                break
        
        if stop_train:
            break
    
    writer.close()




























