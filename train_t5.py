import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from argparse import ArgumentParser
import tqdm
from utils.tools import read_data
from utils.logger import define_logger
from torch.optim import AdamW
import os
import random
import numpy as np
from utils.Dataset import DynamicDataset
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from rouge import Rouge
from nltk import bleu
from nltk.tokenize import word_tokenize




def hyper_parameters():
    parser = ArgumentParser(description='dual learning gpt2')

    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--model_dir', type=str, default='/ssd1/huggingface_transformers/t5-base')
    parser.add_argument('--log_dir', type=str, default='./logger')
    parser.add_argument('--train', type=str, default='train_gen.jsonl')
    parser.add_argument('--test', type=str, default='test_gen.jsonl')
    parser.add_argument('--dev', type=str, default='dev_gen.jsonl')
    parser.add_argument('--output_dir', type=str, default='./output')

    parser.add_argument('--model_name', type=str, default='t5')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--evaluation_steps', type=int, default=200)
    parser.add_argument('--mode', type=str, default='abductive')
    parser.add_argument('--patient', type=int, default=10)
    parser.add_argument('--portion', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=str, default='[3, 0, 1, 2]')
    parser.add_argument('--log_name', type=str, default='bart.txt')
    parser.add_argument('--lambda1', type=float, default=0.99)
    parser.add_argument('--lambda2', type=float, default=0.01)

    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--repetition_penalty', type=float, default=1.5)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3)
    parser.add_argument('--max_len', type=int, default=30)
    parser.add_argument('--seed', type=int, default=3184)

    opt = parser.parse_args()
    return opt


def evaluation(hps, model: T5ForConditionalGeneration, dataloader, tokenizer: T5Tokenizer):
    bleu_score, rougel = 0, 0
    num_instances = 0
    predictions = []
    rouge = Rouge()
    model.eval()

    for batch in dataloader:
        causes, effects = batch
        inputs, outputs = [causes, effects] if hps.mode == 'deductive' else [effects, causes]
        num_instances += len(inputs)

        inputs_t = tokenizer(inputs, padding=True, return_tensors='pt')

        inputs_t = [inputs_t[term].cuda(hps.gpu[0]) for term in inputs_t]

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids = inputs_t[0],
                attention_mask = inputs_t[1],
                do_sample=True,
                repetition_penalty=2.0
            )

        generated_text = tokenizer.batch_decode(generated_ids, clean_up_tokenization_spaces=True,skip_special_tokens=True)
        generated_text = [sent.split('.')[0]+'.' for sent in generated_text]

        for pre, hypo, ref in zip(inputs, generated_text, outputs):
            if hps.mode == 'deductive':
                predictions.append({'cause': pre, 'effect': ref, 'prediction': hypo})
            else:
                predictions.append({'cause': ref, 'effect': pre, 'prediction': hypo})
            
            bleu_score += bleu([word_tokenize(ref)], word_tokenize(hypo), weights=[0.25, 0.25, 0.25, 0.25])
            try:
                rougel += rouge.get_scores(hypo, ref)[0]['rouge-l']['r']
            except:
                continue
    model.train()
    return bleu_score / num_instances, rougel / num_instances, predictions


def tokenization(hps, tokenizer: T5Tokenizer, causes, effects):
    inputs, outputs = [causes, effects] if hps.mode == 'deductive' else [effects, causes]

    inputs_t = tokenizer(inputs, padding=True, return_tensors='pt')
    outputs_t = tokenizer(outputs, padding=True, return_tensors='pt')
    labels = outputs_t.input_ids
    labels[labels==tokenizer.pad_token_id] = -100

    return inputs_t.input_ids, inputs_t.attention_mask, labels, outputs_t.attention_mask


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

    model = T5ForConditionalGeneration.from_pretrained(hps.model_dir)
    ckpt = torch.load('./output/model/t5_{}.ckpt'.format(hps.mode), map_location='cpu')
    model.load_state_dict(ckpt['model_dict'])
    writer = SummaryWriter(log_dir='./tensorboard/{}_{}_portion{}'.format(hps.model_name, hps.mode, round(hps.portion*100)))

    if hps.use_gpu:
        model = model.cuda(hps.gpu[0])

    optimizer = AdamW(model.parameters(), lr=hps.lr)
    optimizer.load_state_dict(ckpt['optimizer_dict'])
    tokenizer = T5Tokenizer.from_pretrained(hps.model_dir)

    train_data = read_data(os.path.join(hps.data_dir, hps.train), portion=hps.portion)
    test_data = read_data(os.path.join(hps.data_dir, hps.test))
    dev_data = read_data(os.path.join(hps.data_dir, hps.dev))

    train = tokenization(hps, tokenizer, train_data[0], train_data[1])

    TRAIN = TensorDataset(*train)
    TEST = DynamicDataset(*test_data)
    DEV = DynamicDataset(*dev_data)
    
    dev_sampler = RandomSampler(DEV, replacement=False, num_samples=500)
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

            batch = [term.cuda(hps.gpu[0]) for term in batch]

            input_ids, attention_mask, labels, _ = batch

            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            loss.backward()
            optimizer.step()

            if steps % 20 == 0:
                writer.add_scalar("{}_loss".format(hps.mode), loss.detach(), steps//20)

            total_loss += loss.item()
            epoch_step += 1
            steps += 1

            bar.set_postfix(loss='{}'.format(total_loss/epoch_step))

        logger.info('[Evaluation] Starting evaluation on dev set')

        dev_scores = evaluation(hps, model, dev_loader, tokenizer)
        logger.info("[Dev {} Rouge-l]: {}".format(hps.mode, dev_scores[1]))
        logger.info('[Dev {} Average Bleu]: {}'.format(hps.mode, dev_scores[0]))
        
        writer.add_scalar("Dev {} Rouge-l".format(hps.mode), dev_scores[1], steps)
        writer.add_scalar("Dev {} Average Bleu".format(hps.mode), dev_scores[0], steps)

        if True:
            output_fi = open('./output/{}_{}.json'.format(hps.model_name, hps.mode), 'w')
            best_score = dev_scores[1]
            patient = 0
            logger.info('[Evaluation] Starting evaluation on test set')

            test_ab = evaluation(hps, model, test_loader, tokenizer)
            logger.info("[Test {} Rouge-l]: {}".format(hps.mode, test_ab[1]))
            logger.info('[Test {} Average Bleu]: {}'.format(hps.mode, test_ab[0]))

            torch.save({'model_dict': model.state_dict(),
                        'optimizer_dict': optimizer.state_dict()},
                        './output/model/{}_{}.ckpt'.format(hps.model_name, hps.mode))

            json.dump(test_ab[-1], output_fi, indent=1)
        
        else:
            patient += 1
            if patient == hps.patient:
                stop_train = True
                break
        
        if stop_train:
            break
    
    writer.close()




























