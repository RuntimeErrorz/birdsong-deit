from optparse import Values
import matplotlib.pyplot as plt
import timm
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
import time
import random
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from skimage.transform import rescale, resize, downscale_local_mean
from tqdm import tqdm
import soundfile as sf
import librosa
import numpy as np
import pandas as pd
import torch
import argparse
import os
import sys
from pathlib import Path
import librosa
import requests
import json

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
input_path = Path('../input/birdclef-2021/')
cpmp_path = Path('../input/cpmp-birdclef21-2/')
PERIOD = 5
IMAGE_HEIGHT = 300
IMAGE_WIDTH = 2*IMAGE_HEIGHT

POSWEIGHT = 8
SR = 32000
torch.__version__
pd.options.display.max_columns = 100
SOUNDLENGTH = []
#from audiomentations import Compose, AddGaussianSNR, AddGaussianNoise, PitchShift, AddBackgroundNoise, AddShortNoises, Gain
LABELS_TO_BIRDS = {}
data = pd.read_csv("labels.csv")
labels = data.label
names = data.name
LABELS_NAMES = {}
for (label, name) in zip(labels, names):
    LABELS_NAMES[label] = name
LABELS_NAMES['nocall'] = 'nocall'


def seed_torch(seed_value):
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
    if torch.backends.cudnn.is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


train = pd.read_csv(cpmp_path / 'train_001.csv')  # actually xeno-canto
train_ff1010 = pd.read_csv(cpmp_path / 'train_ff1010.csv')
train_ff1010['primary_label'] = 'nocall'
columns = ['length', 'primary_label', 'secondary_labels', 'filename']
train = pd.concat((train[columns], train_ff1010[columns])
                  ).reset_index(drop=True)
primary_labels = set(train.primary_label.unique())
secondary_labels = set(
    [s for labels in train.secondary_labels for s in eval(labels)])
secondary_labels - primary_labels
res = [[label for label in eval(secondary_label) if label != 'rocpig1']
       for secondary_label in train['secondary_labels']]
train['secondary_labels'] = res
BIRD_CODE = {}
for i, label in enumerate(sorted(primary_labels)):
    BIRD_CODE[label] = i
INV_BIRD_CODE = np.array(sorted(primary_labels))
NOCALL_CODE = BIRD_CODE['nocall']
NOCALL_CODE
train['class'] = [BIRD_CODE[label] for label in train.primary_label]
df = train.groupby('class').size()
df = 1. / df
df = df / df.mean()
class_weights = df.values
class_weights[BIRD_CODE['nocall']] = 1  # nocall
logits_weights = np.log(class_weights).reshape((1, -1))
device = torch.device('cuda')
SEED = 0
FP16 = False
NFOLDS = 5
TEST_BATCH_SIZE = 32
WORKERS = 2


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=True):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class Backbone(nn.Module):

    def __init__(self, name='resnet18', pretrained=False, deit_token=None, in_chans=3):
        super(Backbone, self).__init__()
        self.net = timm.create_model(
            name, pretrained=pretrained, in_chans=in_chans)
        if 'regnet' in name:
            self.out_features = self.net.head.fc.in_features
        elif name == 'vit_deit_base_distilled_patch16_384' and deit_token == 'cat':
            self.out_features = self.net.head.in_features + self.net.head_dist.in_features
        elif 'vit' in name:
            self.out_features = self.net.head.in_features
        elif 'nfnet' in name:
            self.out_features = self.net.head.fc.in_features
        elif 'swin' in name:
            self.out_features = self.net.head.in_features
        elif 'rexnet' in name:
            self.out_features = self.net.head.fc.in_features
        elif 'csp' in name:
            self.out_features = self.net.head.fc.in_features
        elif 'res' in name:  # works also for resnest
            self.out_features = self.net.fc.in_features
        elif 'efficientnet' in name:
            self.out_features = self.net.classifier.in_features
        elif 'densenet' in name:
            self.out_features = self.net.classifier.in_features
        elif 'senet' in name:
            self.out_features = self.net.fc.in_features
        elif 'inception' in name:
            self.out_features = self.net.last_linear.in_features

        else:
            self.out_features = self.net.classifier.in_features

    def forward(self, x):
        x = self.net.forward_features(x)

        return x


class BirdModel(nn.Module):
    def __init__(self, backbone, out_dim, neck=None, embedding_size=512, gem_pooling=False,
                 loss=False, pretrained=False, use_pos=True, deit_token=None, in_chans=3):
        super(BirdModel, self).__init__()
        self.backbone_name = backbone
        self.loss = loss
        self.embedding_size = embedding_size
        self.out_dim = out_dim
        self.use_pos = use_pos
        self.deit_token = deit_token
        self.in_chans = in_chans
        self.backbone = Backbone(
            backbone, pretrained=pretrained, deit_token=deit_token, in_chans=in_chans)

        if gem_pooling == "gem":
            self.global_pool = GeM(p_trainable=args.p_trainable)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        # https://www.groundai.com/project/arcface-additive-angular-margin-loss-for-deep-face-recognition
        if neck == "option-D":
            self.neck = nn.Sequential(
                nn.Linear(self.backbone.out_features,
                          self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        elif neck == "option-F":
            self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.backbone.out_features,
                          self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        else:
            self.neck = nn.Sequential(
                nn.Linear(self.backbone.out_features,
                          self.embedding_size, bias=False),
                nn.BatchNorm1d(self.embedding_size),
            )

        self.head = nn.Linear(self.embedding_size, out_dim)

    def forward(self, input_dict, get_embeddings=False, get_attentions=False):

        x = input_dict['spect']
        x = x.unsqueeze(1)
        if self.use_pos:
            pos = torch.linspace(0., 1., x.size(2)).to(x.device)
            pos = pos.half()
            pos = pos.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            pos = pos.expand(x.size(0), 1, x.size(2), x.size(3))
            if self.in_chans == 2:
                x = x.expand(-1, 1, -1, -1)
                x = torch.cat([x, pos], 1)
            else:
                x = x.expand(-1, 2, -1, -1)
                x = torch.cat([x, pos], 1)
        else:
            x = x.expand(-1, 3, -1, -1)

        x = self.backbone(x)

        if 'vit' not in self.backbone_name and 'swin' not in self.backbone_name:
            x = self.global_pool(x)
            x = x[:, :, 0, 0]
        if 'vit_deit_base_distilled_patch16_384' == self.backbone_name:
            if self.deit_token == 'sum':
                x = x[0] + x[1]
            elif self.deit_token == 'cat':
                x = torch.cat(x, 1)
            else:
                x = x[self.deit_token]

        x = self.neck(x)

        logits = self.head(x)

        output_dict = {'logits': logits,
                       }
        if self.loss:
            target = input_dict['target']
            secondary_mask = input_dict['secondary_mask']
            loss = criterion(logits, target, secondary_mask)

            output_dict['loss'] = loss

        return output_dict


def load_checkpoint(backbone, epoch, fold, seed, fname, use_pos, deit_token):
    filepath = cpmp_path / ('%s_%d_%d_%d.pt' % (fname, fold, seed, epoch))
    print('loading ', str(filepath), '...')
    model = BirdModel(backbone,
                      out_dim=len(BIRD_CODE),
                      neck="option-F",
                      loss=False,
                      gem_pooling=False,
                      use_pos=use_pos,
                      deit_token=deit_token,
                      pretrained=False).to(device)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    model.half()
    return model


def list_files(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


IS_TEST = True
test_audio = list_files(input_path / 'test_soundscapes')
if len(test_audio) == 0:
    test_audio = list_files(input_path / 'train_soundscapes')
    IS_TEST = False

print('{} FILES IN TEST SET.'.format(len(test_audio)))


def get_clip_sr(file_path):
    clip, sr_native = sf.read(file_path)
    clip = librosa.to_mono(clip)
    clip = clip.astype('float32')
    sr = 32000
    return clip, sr


def get_melspec(clip, sr, period, IMAGE_WIDTH, IMAGE_HEIGHT, fmin, htk, power, n_fft):
    length = len(clip)
    if period > length:
        start = np.random.randint(period - length)
        tmp = np.zeros(period, dtype=clip.dtype)
        tmp[start: start + length] = clip
        clip = tmp

    win_length = n_fft  # //2
    hop_length = int((len(clip) - win_length + n_fft) / IMAGE_WIDTH) + 1
    spect = np.abs(librosa.stft(y=clip, n_fft=n_fft,
                   hop_length=hop_length, win_length=win_length))
    if spect.shape[1] < IMAGE_WIDTH:
        #print('too large hop length, len(clip)=', len(clip))
        hop_length = hop_length - 1
        spect = np.abs(librosa.stft(y=clip, n_fft=n_fft,
                       hop_length=hop_length, win_length=win_length))
    if spect.shape[1] > IMAGE_WIDTH:
        spect = spect[:, :IMAGE_WIDTH]
    n_mels = IMAGE_HEIGHT // 2
    spect = np.power(spect, power)
    spect = librosa.feature.melspectrogram(
        S=spect, sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=16000, htk=htk)
    spect = librosa.power_to_db(spect)
    spect = resize(spect, (IMAGE_HEIGHT, IMAGE_WIDTH),
                   preserve_range=True, anti_aliasing=True)
    spect = spect - spect.min()
    smax = spect.max()
    if smax >= 0.001:
        spect = spect / smax
    else:
        spect[...] = 0
    return spect


class BirdTestDataset(Dataset):
    def __init__(self,
                 length,
                 file_path,
                 image_width,
                 image_height,
                 period=PERIOD,
                 fmin=300,
                 htk=False,
                 power=2,
                 use_patch=False,
                 n_fft=1024,
                 single_window=False,
                 use_inv_stem=False,
                 ):
        super(BirdTestDataset, self).__init__()
        self.file_path = file_path
        clip, sr = get_clip_sr(file_path)
        self.clip = clip
        self.sr = sr
        self.image_width = image_width
        self.image_height = image_height
        self.fmin = fmin
        self.htk = htk
        self.power = power
        self.use_patch = use_patch
        self.n_fft = n_fft
        self.single_window = single_window
        self.use_inv_stem = use_inv_stem
        period = period * sr
        self.period = period
        #print(clip.shape[0] / (period))
        if single_window:
            self.starts = np.arange(0, length + 2.5, 2.5)
        else:
            self.starts = np.arange(0, length + 1.25, 1.25)
        #self.starts = np.arange(0, 602.5, 2.5)
        # print(self.starts)

    def __len__(self):
        if self.single_window:
            return len(self.starts) - 2
        else:
            return len(self.starts) - 4

    def inv_stem(self, x):
        x1 = x.transpose(0, 1).view(24, 24, 16, 16)
        y = torch.zeros(384, 384, dtype=x.dtype)
        for i in range(24):
            for j in range(24):
                y[i*16:(i+1)*16, j*16:(j+1)*16] = x1[i, j]
        return y

    def __getitem__(self, idx: int):
        start = self.starts[idx]
        if self.single_window:
            end = self.starts[idx + 2]
        else:
            end = self.starts[idx + 4]

        clip = self.clip[int(start * self.sr): int(end * self.sr)]
        melspec = get_melspec(clip, self.sr, self.period, self.image_width, self.image_height, self.fmin,
                              self.htk, self.power, self.n_fft)

        if self.use_inv_stem:
            spect = torch.from_numpy(melspec)
            spect = self.inv_stem(spect)
        else:
            if self.use_patch:
                patch_size = self.use_patch
                spect = np.zeros((384, 384), dtype=np.float32)
                for i in range(0, 192, patch_size):
                    spect[2 * i: 2 * i + patch_size,
                          :] = melspec[i: i + patch_size, : 384]
                    spect[2 * i + patch_size: 2 * i + 2*patch_size,
                          :] = melspec[i: i + patch_size, 384:]
                melspec = spect
            spect = torch.from_numpy(melspec)

        return {
            "spect": spect.half(),
        }


def test_preds(loader, models, device):
    for model in models:
        model.eval()
    LOGITS = []

    with torch.no_grad():
        if 1:
            bar = (range(len(loader)))
            load_iter = iter(loader)
            batch = load_iter.next()
            batch = {k: batch[k].to(device, non_blocking=True)
                     for k in batch.keys()}

            for i in bar:
                input_dict = batch.copy()
                if i + 1 < len(loader):
                    batch = load_iter.next()
                    batch = {k: batch[k].to(device, non_blocking=True)
                             for k in batch.keys()}

                logits = 0
                for model in models:
                    logits = logits + model(input_dict)['logits'].detach()
                logits = logits / len(models)
                LOGITS.append(logits)

    LOGITS = torch.cat(LOGITS).cpu().numpy()

    return LOGITS


def window(logits):
    new_logits = 2*logits[0::2, :]
    median_logits = logits[1::2, :]
    new_logits[:-1] += median_logits
    new_logits[1:] += median_logits
    new_logits[:1, :] += logits[:1, :]
    new_logits[-1:, :] += logits[-1:, :]
    new_logits = new_logits / 4
    return new_logits


def compute_logits(config, models):
    LOGITS = []
    RAW_LOGITS = []
    for file_path in tqdm(test_audio):
        fmin = config.get('fmin', 300)
        htk = config.get('htk', False)
        power = config.get('power', 2)
        use_patch = config.get('use_patch', False)
        n_fft = config.get('n_fft', 1024)
        single_window = config.get('single_window', False)
        use_inv_stem = config.get('use_inv_stem', False)
        SOUNDLENGTHITEM = int(librosa.get_duration(filename=file_path))
        SOUNDLENGTH.append(SOUNDLENGTHITEM)
        test_dataset = BirdTestDataset(SOUNDLENGTHITEM, file_path, config['width'], config['height'],
                                       fmin=fmin, htk=htk, power=power, use_patch=use_patch,
                                       n_fft=n_fft, single_window=single_window, use_inv_stem=use_inv_stem,)
        test_loader = DataLoader(
            test_dataset,
            batch_size=TEST_BATCH_SIZE,
            num_workers=WORKERS,
            shuffle=False,
        )
        logits = test_preds(test_loader, models, device)
        logits = window(logits)
        if not single_window:
            logits = window(logits)
        LOGITS.append(logits)

    return LOGITS


def compute_pred(LOGITS, thr, incr, weight=0):
    all_preds = []
    index = 0
    for file_path, logits in zip(test_audio, LOGITS):
        createTime = os.path.getctime(file_path)
        audio_id = file_path.split(os.sep)[-1].split('.')[0]
        logits = logits + weight * logits_weights
        logits_max = logits.max(0)  # 最大的一行
        logits = logits.copy()
        for j in range(logits.shape[1]):
            if logits_max[j] > thr:
                logits[:, j] += thr - incr

        for seconds, logit in zip(range(5, SOUNDLENGTH[index], 5), logits):
            birds = list(INV_BIRD_CODE[logit >= thr])
            if 'nocall' in birds and len(birds) > 1:
                birds = [p for p in birds if p != 'nocall']
            elif len(birds) == 0:
                birds = ['nocall']
            birds = ' '.join(birds)
            birdList = birds.split(' ')
            values = ''
            for i in birdList:
                values += LABELS_NAMES[i]
                values += '    '
            values = values.strip()
            all_preds.append((str(audio_id), seconds, values,createTime))
        index += 1
    return all_preds


config = {'fnames': ['stft_transformer', ],
          'backbone': 'vit_deit_base_distilled_patch16_384',
          'width': 576,
          'height': 256,
          'use_pos': False,
          'deit_token': 'sum',
          'use_inv_stem': True,
          }

epoch = 59
deit_token = config.get('deit_token', None)
models = [load_checkpoint(config['backbone'], epoch, fold, seed,
                          fname + '_' + str(fold),
                          config['use_pos'],
                          deit_token,
                          )
          for fold in range(NFOLDS)
          for seed, fname in enumerate(config['fnames'])]

thr = 2.7
incr = -1.1
LOGITS = compute_logits(config, models)
datas = compute_pred(LOGITS, thr, incr)
id = -1
with open("id.txt","r+") as i:
    id = int(i.readline().strip('\n'))
    for data in datas:
        item = {'id': id, 'filename': data[0], 'time': data[1], 'bird_v': data[2]}
        requests.post(url="http://120.24.50.133:8080/bird",
                    data=json.dumps(item), headers={"Content-Type": "application/json"})
        id += 1
    i.seek(0)
    i.write(str(id))
