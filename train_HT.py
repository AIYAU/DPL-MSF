import numpy as np
import os
import argparse
import pickle
import time
import imp
import logging
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from model.mapping import Mapping
from model.encoder import Encoder
from model.encoder import PrototypeGenerator
from utils.dataloader import get_HBKC_data_loader, Task, get_target_dataset, tagetSSLDataset
from utils import utils, loss_function, data_augment

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from peft import PromptTuningConfig, get_peft_model, TaskType


parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument('--config', type=str, default=os.path.join( './config', 'HT.py'))
args = parser.parse_args()

# load hyperparameters
config = imp.load_source("", args.config).config
train_opt = config['train_config']
data_path = config['data_path']
source_data = config['source_data']
target_data = config['target_data']
target_data_gt = config['target_data_gt']
log_dir = config['log_dir']
patch_size = train_opt['patch_size']
batch_task = train_opt['batch_task']
emb_size = train_opt['d_emb']
SRC_INPUT_DIMENSION = train_opt['src_input_dim']
TAR_INPUT_DIMENSION = train_opt['tar_input_dim']
N_DIMENSION = train_opt['n_dim']
SHOT_NUM_PER_CLASS = train_opt['shot_num_per_class']
QUERY_NUM_PER_CLASS = train_opt['query_num_per_class']
EPISODE = train_opt['episode']
LEARNING_RATE = train_opt['lr']
GPU = config['gpu']
TAR_CLASS_NUM = train_opt['tar_class_num'] # the number of class
TAR_LSAMPLE_NUM_PER_CLASS = train_opt['tar_lsample_num_per_class'] # the number of labeled samples per class
WEIGHT_DECAY = train_opt['weight_decay']

utils.same_seeds(0)

# get src/tar class number -> label semantic vector
labels_src = ["water", "bare soil school", "bare soil park", "bare soil farmland", "natural plants", "weeds in farmland", "forest", "grass", "rice field grown", "rice field first stage", "row crops", "plastic house", "manmade non dark", "manmade dark", "manmade blue", "manmade red", "manmade grass", "asphalt"]

# # IP
# labels_tar = ["Alfalfa", "Corn notill", "Corn mintill", "Corn", "Grass pasture", "Grass trees", "Grass pasture mowed", "Hay windrowed", "Oats", "Soybean notill", "Soybean mintill", "Soybean clean", "Wheat", "Woods", "Buildings Grass Trees Drives", "Stone Steel Towers"]

# houston
labels_tar = ["Healthy grass", "Stressed grass", "Synthetic grass", "Trees", "Soil", "Water", "Residential", "Commercial", "Road", "Highway", "Railway", "Parking Lot 1", "Parking Lot 2", "Tennis Court", "Running Track"]

# salinas
# labels_tar = ["Brocoli green weeds 1", "Brocoli green weeds 2", "Fallow", "Fallow rough plow", "Fallow smooth", "Stubble", "Celery", "Grapes untrained", "Soil vinyard develop", "Corn senesced green weeds","Lettuce romaine 4wk", "Lettuce romaine 5wk", "Lettuce romaine 6wk", "Lettuce romaine 7wk" , "Vinyard untrained", "Vinyard vertical trellis"]

# UP
# labels_tar = ["Asphalt", "Meadows", "Gravel", "Tress", "Sheets", "Bare soil", "Bitumen", "Bricks", "Shadow"]


'''加载预训练模型Bert'''
##################################################################################################
from transformers import BertModel, BertTokenizer
model = BertModel.from_pretrained('pretrain-model/bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('pretrain-model/bert-base-uncased')
config = PromptTuningConfig(task_type=TaskType.FEATURE_EXTRACTION , num_virtual_tokens=2)
##################################################################################################



'''加载源域数据集'''
##################################################################################################
with open(os.path.join(data_path, source_data), 'rb') as handle:
    source_imdb = pickle.load(handle)

data_train = source_imdb['data']
labels_train = source_imdb['Labels']

keys_all_train = sorted(list(set(labels_train)))
label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
train_dict = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_dict:
        train_dict[label_encoder_train[class_]] = []
    train_dict[label_encoder_train[class_]].append(path)
del keys_all_train
del label_encoder_train

metatrain_data = utils.sanity_check(train_dict)

for class_ in metatrain_data:
    for i in range(len(metatrain_data[class_])):
        metatrain_data[class_][i] = np.transpose(metatrain_data[class_][i], (2, 0, 1))
##################################################################################################



# 加载目标域数据集
test_data = os.path.join(data_path,target_data)
test_label = os.path.join(data_path,target_data_gt)
Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)


# 损失函数的设计--初始化
crossEntropy = nn.CrossEntropyLoss().to(GPU)
cos_criterion = nn.CosineSimilarity(dim=1).to(GPU)
infoNCE_Loss = loss_function.ContrastiveLoss(batch_size=TAR_CLASS_NUM).to(GPU)
infoNCE_Loss_SSL = loss_function.ContrastiveLoss(batch_size=128).to(GPU)
SupConLoss_t = loss_function.SupConLoss(temperature=0.1).to(GPU)


# 实验结果指标
nDataSet = 10
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, TAR_CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None,None,None,None,None

seeds = [1236, 1237, 1226, 1227, 1211, 1212, 1216, 1240, 1222, 1223]

# 日志设置
experimentSetting = '{}way_{}shot_{}'.format(TAR_CLASS_NUM, TAR_LSAMPLE_NUM_PER_CLASS, target_data.split('/')[0])
utils.set_logging_config(os.path.join(log_dir, experimentSetting), nDataSet)
logger = logging.getLogger('main')
logger.info('seeds_list:{}'.format(seeds))

for iDataSet in range(nDataSet):
    logger.info('emb_size:{}'.format(emb_size))
    logger.info('patch_size:{}'.format(patch_size))
    logger.info('seeds:{}'.format(seeds[iDataSet]))

    utils.same_seeds(seeds[iDataSet])

    # 加载目标域数据集来进行训练和测试
    train_loader, test_loader, target_da_metatrain_data, G, RandPerm, Row, Column,nTrain, target_aug_data_ssl, target_aug_label_ssl = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler,
        GroundTruth=GroundTruth,
        class_num=TAR_CLASS_NUM,
        tar_lsample_num_per_class=TAR_LSAMPLE_NUM_PER_CLASS,
        shot_num_per_class=TAR_LSAMPLE_NUM_PER_CLASS,
        patch_size=patch_size)

    target_ssl_dataset = tagetSSLDataset(target_aug_data_ssl, target_aug_label_ssl)
    target_ssl_dataloader = torch.utils.data.DataLoader(target_ssl_dataset, batch_size=64, shuffle=True, drop_last=True)

    num_supports, num_samples, query_edge_mask, evaluation_mask = utils.preprocess(TAR_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS, batch_task, GPU)
    
    # 实例化模型
    mapping_src = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION).to(GPU)
    mapping_tar = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION).to(GPU)
    encoder = Encoder(n_dimension=N_DIMENSION, patch_size=patch_size, emb_size=emb_size, dropout=0.3).to(GPU)
    protoGenerator = PrototypeGenerator().to(GPU)
    P_model = get_peft_model(model, config).to(GPU)

    
    # 优化器-学习率设置
    mapping_src_optim = torch.optim.Adam(mapping_src.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    mapping_tar_optim = torch.optim.Adam(mapping_tar.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    P_model_optim = torch.optim.Adam(P_model.parameters(), lr=LEARNING_RATE)
    protoGenerator_optim = torch.optim.Adam(protoGenerator.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


    # 权重初始化
    mapping_src.apply(utils.weights_init).to(GPU)
    mapping_tar.apply(utils.weights_init).to(GPU)
    encoder.apply(utils.weights_init).to(GPU)
    protoGenerator.apply(utils.weights_init).to(GPU)

    

    # 设置训练模型
    mapping_src.train()
    mapping_tar.train()
    encoder.train()
    P_model.train()
    protoGenerator.train()

    logger.info("Training...")
    last_accuracy = 0.0
    best_episode = 0
    total_hit_src, total_num_src, total_hit_tar, total_num_tar, acc_src, acc_tar = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    train_start = time.time()
    writer = SummaryWriter()

    target_ssl_iter = iter(target_ssl_dataloader)

    for episode in range(EPISODE):


        '''---在Task任务中进行采样,划分,进行小样本学习任务---'''
        ##############################################################################
        task_src = Task(metatrain_data, TAR_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
        support_dataloader_src = get_HBKC_data_loader(task_src, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
        query_dataloader_src = get_HBKC_data_loader(task_src, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=False)
        
        task_tar = Task(target_da_metatrain_data, TAR_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
        support_dataloader_tar = get_HBKC_data_loader(task_tar, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
        query_dataloader_tar = get_HBKC_data_loader(task_tar, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=False)
        ##############################################################################
       
       
        '''---获取真实采样的Label---'''
        ##############################################################################
        support_src, support_label_src = next((iter(support_dataloader_src)))
        query_src, query_label_src = next((iter(query_dataloader_src)))
        support_real_labels_src = task_src.support_real_labels  #源域label
        support_real_labels_tar = task_tar.support_real_labels  #目标域label
        #############################################################################

        
        '''---目标域的hard_label---'''
        ##############################################################################
        #源域19个类别,所以总长度设置为19,虽是16-way-1-shot训练,但是采样会大于16类
        hard_label_src = np.zeros((len(support_real_labels_src), 19)) 
        #目标域16个类别,后面的补足0,0不代表含义,所以可以补零操作,保证维度一致
        hard_label_tar = np.zeros((len(support_real_labels_tar), 19)) 

        for i, num in enumerate(support_real_labels_src):
            hard_label_src[i][num] = 1

        for i, num in enumerate(support_real_labels_tar):
            hard_label_tar[i][num] = 1
        
        #源域hard_label
        hard_label_src = torch.from_numpy(hard_label_src).float()
        hard_label_src = hard_label_src.cuda(GPU)

        #目标域hard_label
        hard_label_tar = torch.from_numpy(hard_label_tar).float()
        hard_label_tar = hard_label_tar.cuda(GPU)
        ##############################################################################



        '''---加入高斯扰动,形成源域与目标域soft_label---'''
        ##############################################################################
        # 独热编码的长度
        num_classes = 19
        # 生成一个高斯分布扰动的张量
        std_dev = 0.010     # 标准差
        mean_value = 0   # 均值
        perturbation = torch.randn(num_classes) * std_dev + mean_value # 生成扰动 
        perturbation = perturbation.abs() # 取扰动的绝对值，确保所有值都是正数
        perturbation_src = perturbation.to('cuda:0') 
        perturbation_tar = perturbation.to('cuda:0') 

        # 确保原始类别依旧值最大
        # 源域
        original_index_src = torch.argmax(hard_label_src).item()
        max_perturbation_src = perturbation_src.max()
        perturbation_src[original_index_src] = max_perturbation_src
        
        # 目标域
        original_index_tar = torch.argmax(hard_label_tar).item()
        max_perturbation_tar = perturbation_tar.max()
        perturbation_tar[original_index_tar] = max_perturbation_tar

        # 将扰动加到原始的独热编码上
        soft_labels_tar = hard_label_tar + perturbation_tar
        soft_labels_src = hard_label_src + perturbation_src

        # 归一化，确保所有元素的和为1
        soft_labels_src = soft_labels_src / soft_labels_src.sum(dim=1, keepdim=True)
        soft_labels_tar = soft_labels_tar / soft_labels_tar.sum(dim=1, keepdim=True)
        ##############################################################################



               
        """提取文本特征"""
        ##############################################################################
        encoded_inputs_src = tokenizer(labels_src, padding=True, truncation=True, return_tensors='pt').to(GPU)
        input_ids_src = encoded_inputs_src['input_ids']
        attention_mask_src = encoded_inputs_src['attention_mask'] 
        with torch.no_grad():
            outputs_src = P_model(input_ids=input_ids_src, attention_mask=attention_mask_src)
        semantic_mapping_src = outputs_src.last_hidden_state[:, 0, :]  # (num_classess, 768)

       
        encoded_inputs_tar = tokenizer(labels_tar, padding=True, truncation=True, return_tensors='pt').to(GPU)
        input_ids_tar = encoded_inputs_tar['input_ids']
        attention_mask_tar = encoded_inputs_tar['attention_mask']         
        with torch.no_grad():
            outputs_tar = P_model(input_ids=input_ids_tar, attention_mask=attention_mask_tar)
        semantic_mapping_tar = outputs_tar.last_hidden_state[:, 0, :]  # (num_classess, 768)

        semantic_mapping_src = semantic_mapping_src.cpu().numpy()
        semantic_mapping_tar = semantic_mapping_tar.cpu().numpy()
        ##############################################################################
        


        '''---根据采样的Label来获取文本特征---'''
        ##############################################################################
        semantic_support_src = torch.zeros(TAR_CLASS_NUM, 768)
        for i, class_id in enumerate(support_real_labels_src) :
            semantic_support_src[i] = torch.from_numpy(semantic_mapping_src[class_id])

        semantic_support_tar = torch.zeros(TAR_CLASS_NUM, 768)
        for i, class_id in enumerate(support_real_labels_tar):
            semantic_support_tar[i] = torch.from_numpy(semantic_mapping_tar[class_id])
        ##############################################################################



        '''---根据采样的Label来获取源域&目标域图像样本---'''
        ##############################################################################
        support_tar, support_label_tar = next(iter(support_dataloader_tar))
        query_tar, query_label_tar = next(iter(query_dataloader_tar))
        ##############################################################################



        '''---特征提取-获取源域文本特征&图像特征---'''
        ##############################################################################
        support_features_src, semantic_feature_src = encoder(mapping_src(support_src.to(GPU)), semantic_feature=semantic_support_src.to(GPU), s_or_q = "support") # (9, 160)
        query_features_src = encoder(mapping_src(query_src.to(GPU)))
        ##############################################################################



        
        '''---特征提取-获取目标域文本特征&图像特征---'''
        ##############################################################################
        support_features_tar, semantic_feature_tar = encoder(mapping_tar(support_tar.to(GPU)), semantic_feature=semantic_support_tar.to(GPU), s_or_q = "support")  # (9, 160)
        query_features_tar = encoder(mapping_tar(query_tar.to(GPU)))
        ##############################################################################


        
        '''---若训练样本>1, 则取平均求原型---'''
        ##############################################################################
        if SHOT_NUM_PER_CLASS > 1:
            support_proto_src = support_features_src.reshape(TAR_CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
            support_proto_tar = support_features_tar.reshape(TAR_CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)

        else:
            support_proto_src = support_features_src
            support_proto_tar = support_features_tar
        ##############################################################################




        '''---Norm化图像原型---''' 
        ##############################################################################
        norm_image_src = support_proto_src.norm(dim=1, keepdim=True)
        norm_support_proto_src = support_proto_src / norm_image_src

        norm_iamge_tar = support_proto_tar.norm(dim=1, keepdim=True)
        norm_support_proto_tar = support_proto_tar / norm_iamge_tar
        ##############################################################################


        '''---Norm查询集---''' 
        ##############################################################################
        norm_query_src = query_features_src.norm(dim=1, keepdim=True)
        norm_query_features_src = query_features_src / norm_query_src

        norm_query_tar = query_features_tar.norm(dim=1, keepdim=True)
        norm_query_features_tar = query_features_tar / norm_query_tar
        ##############################################################################



        '''Norm原型&soft label修正原型'''
        ##############################################################################
        support_proto_sl_src = torch.cat([norm_support_proto_src, soft_labels_src], dim=1).to(GPU)
        support_proto_sl_tar = torch.cat([norm_support_proto_tar, soft_labels_tar], dim=1).to(GPU)     
        support_proto_sl_src = protoGenerator(support_proto_sl_src)
        support_proto_sl_tar = protoGenerator(support_proto_sl_tar)
        ##############################################################################
        


        '''---计算logits&小样本损失---''' 
        ##############################################################################
        logits_src = utils.euclidean_metric(norm_query_features_src, support_proto_sl_src)
        logits_tar = utils.euclidean_metric(norm_query_features_tar, support_proto_sl_tar)

        
        f_loss_src = crossEntropy(logits_src, query_label_src.long().to(GPU))
        f_loss_tar = crossEntropy(logits_tar, query_label_tar.long().to(GPU))

        
        f_loss = f_loss_src + f_loss_tar 
        
        text_align_loss = infoNCE_Loss(semantic_feature_src, support_proto_src) + infoNCE_Loss(semantic_feature_tar, support_proto_tar)
        ##############################################################################



        '''总损失'''
        ##############################################################################
        loss = f_loss + 2.5 * text_align_loss
        ##############################################################################



        mapping_src.zero_grad()
        mapping_tar.zero_grad()
        P_model.zero_grad()
        encoder.zero_grad()
        protoGenerator.zero_grad()

        loss.backward()

        mapping_src_optim.step()
        P_model_optim.step()
        mapping_tar_optim.step()
        encoder_optim.step()
        protoGenerator_optim.step()

        total_hit_src += torch.sum(torch.argmax(logits_src, dim=1).cpu() == query_label_src).item()
        total_num_src += query_src.shape[0]
        acc_src = total_hit_src / total_num_src

        total_hit_tar += torch.sum(torch.argmax(logits_tar, dim=1).cpu() == query_label_tar).item()
        total_num_tar += query_tar.shape[0]
        acc_tar = total_hit_tar / total_num_tar

        if (episode + 1) % 100 == 0:
            logger.info('episode: {:>3d}, f_loss: {:6.4f}, text_align_loss: {:6.4f}, loss: {:6.4f}, acc_src: {:6.4f}, acc_tar: {:6.4f}'.format(
                episode + 1,
                f_loss.item(),
                text_align_loss.item(),
                # scl_loss_tar.item(),
                loss.item(),
                acc_src,
                acc_tar))

            writer.add_scalar('Loss/f_loss', f_loss.item(), episode + 1)
            writer.add_scalar('Loss/text_align_loss', text_align_loss.item(), episode + 1)
            # writer.add_scalar('Loss/scl_loss_tar', scl_loss_tar.item(), episode + 1)
            writer.add_scalar('Loss/loss', loss.item(), episode + 1)

            writer.add_scalar('Acc/acc_src', acc_src, episode + 1)
            writer.add_scalar('Acc/acc_tar', acc_tar, episode + 1)

        if (episode + 1) % 500 == 0 or episode == 0:
            with torch.no_grad():
                # test
                logger.info("Testing ...")
                train_end = time.time()
                mapping_tar.eval()
                encoder.eval()
                total_rewards = 0
                counter = 0
                accuracies = []
                predict = np.array([], dtype=np.int64)
                predict_gnn = np.array([], dtype=np.int64)
                labels = np.array([], dtype=np.int64)

                train_datas, train_labels = next(iter(train_loader))

                support_real_labels = train_labels

                semantic_support = torch.zeros(TAR_CLASS_NUM*TAR_LSAMPLE_NUM_PER_CLASS, 768)
                for i, class_id in enumerate(support_real_labels):
                    semantic_support[i] = torch.from_numpy(semantic_mapping_tar[class_id])

                train_features, _ = encoder(mapping_tar(Variable(train_datas).to(GPU)), semantic_feature = semantic_support.to(GPU),  s_or_q = "support")

                max_value = train_features.max()
                min_value = train_features.min()
                print(max_value.item())
                print(min_value.item())
                train_features = (train_features - min_value) * 1.0 / (max_value - min_value)


                KNN_classifier = KNeighborsClassifier(n_neighbors=1)
                KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)

                # ########################################################################
                # all_test_features = []   ####存贮所有测试集的特征  
                # all_test_labels = []   ####存贮所有测试集对应的标签   
                # batch_counter = 0  # 添加一个计数器来跟踪批次数量     
                # ########################################################################
                for test_datas, test_labels in test_loader:
                    batch_size = test_labels.shape[0]

                    test_features = encoder(mapping_tar((Variable(test_datas).to(GPU))))
                    test_features = (test_features - min_value) * 1.0 / (max_value - min_value)


                    # ######################################################################
                    # if batch_counter <= 140:
                    #     all_test_features.append(test_features)
                    #     all_test_labels.append(test_labels)

                    # batch_counter += 1  # 每次迭代后增加计数器
                    # ###########################################################################

                    predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
                    test_labels = test_labels.numpy()
                    rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)
                    counter += batch_size

                    predict = np.append(predict, predict_labels)
                    labels = np.append(labels, test_labels)

                    accuracy = total_rewards / 1.0 / counter
                    accuracies.append(accuracy)

                test_accuracy = 100. * total_rewards / len(test_loader.dataset)
                writer.add_scalar('Acc/acc_test', test_accuracy, episode + 1)

                logger.info('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset), 100. * total_rewards / len(test_loader.dataset)))
                
                ###################################################################################
                # all_test_features = torch.stack(all_test_features).view(-1,128)  #128表示特征最终的输出维度
                # all_test_labels = torch.stack(all_test_labels).view(-1)
                # print(all_test_features.shape)
                # print(all_test_labels.shape)
                # ###################################################################################

                test_accuracy = 100. * total_rewards / len(test_loader.dataset)

                print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format( total_rewards, len(test_loader.dataset),
                    100. * total_rewards / len(test_loader.dataset)))
                
                ###################################################################################
                # # 使用t-SNE进行降维
                # tsne = TSNE(n_components=2, random_state=42)
                # X_tsne = tsne.fit_transform(all_test_features.cpu().detach().numpy())

                # # 归一化
                # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
                # X_norm = (X_tsne - x_min) / (x_max - x_min)

                # # 绘制t-SNE可视化图
                # plt.figure(figsize=(10, 8))
                # plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 图中文字体设置为Times New Roman

                # shape_list = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
                # color_list = [(0.77, 0.87, 0.7), (0.43, 0.67, 0.27), (0.32, 0.50, 0.2), (0.21, 0.34, 0.13), (0.77, 0.35, 0.06), (0, 0.69, 0.94), (0.75, 0, 0),
                #           (0.7, 0.78, 0.9), (0.48, 0.48, 0.48), (0.95, 0.69, 0.51), (0.97, 0.79, 0.67), (0.34, 0.34, 0.34), (0.8, 0.8, 0),
                #           (0, 0.8, 0.4), (1, 0, 0)]  # 设置不同类别的颜色，避免重复


                # label_list = ['1', '2', '3', '4', '5', '6', '7', '8','9','10','11','12','13','14','15']
                # # 遍历所有标签种类
                # for i in range(len(np.unique(all_test_labels))):
                #     plt.scatter(X_norm[all_test_labels == i, 0], X_norm[all_test_labels == i, 1],color=color_list[i % len(color_list)],
                #                 marker=shape_list[i % len(shape_list)], s=80, label=label_list[i])

                # # 添加图例，并设置字体大小################################################
                # plt.legend(fontsize=15,loc='upper right',bbox_to_anchor=(1.13, 1.02))

                # ax = plt.gca()  # gca:get current axis得到当前轴
                # ax.spines['right'].set_linewidth(2.0)  # 设置边框线宽为2.0
                # ax.spines['top'].set_linewidth(2.0)  # 设置边框线宽为2.0
                # ax.spines['bottom'].set_linewidth(2.0)  # 设置边框线宽为2.0
                # ax.spines['left'].set_linewidth(2.0)  # 设置边框线宽为2.0

                # plt.xticks(fontsize=20)  # 定义坐标轴刻度
                # plt.yticks(fontsize=20)


                # plt.axis('equal')  # 确保x和y轴的缩放相同

                # # plt.show()  # 显示图形
                # plt.savefig(f'./TSNE/HT/HT_visualization_seed_{seeds[iDataSet]}_episode_{episode+1}_acc{test_accuracy}.png', dpi=600)
                ##################################################################################                
                
                test_end = time.time()


                mapping_tar.train()
                encoder.train()
                if test_accuracy > last_accuracy:
                    last_accuracy = test_accuracy
                    best_episode = episode
                    acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                    OA = acc
                    C = metrics.confusion_matrix(labels, predict)
                    A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=float)
                    best_predict_all = predict
                    best_G, best_RandPerm, best_Row, best_Column, best_nTrain = G, RandPerm, Row, Column, nTrain
                    k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

                logger.info('best episode:[{}], best accuracy={}'.format(best_episode + 1, last_accuracy))

    logger.info('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episode + 1, last_accuracy))
    logger.info ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
    logger.info("accuracy list: {}".format(acc))
    logger.info('***********************************************************************************')


OAMean = np.mean(acc)
OAStd = np.std(acc)

AA = np.mean(A, 1)
AAMean = np.mean(AA,0)
AAStd = np.std(AA)

kMean = np.mean(k)
kStd = np.std(k)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

logger.info ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
logger.info ("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
logger.info ("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format( OAStd))
logger.info ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
logger.info ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
logger.info ("accuracy list: {}".format(acc))
logger.info ("accuracy for each class: ")
for i in range(TAR_CLASS_NUM):
    logger.info ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))


#################classification map################################
for i in range(len(best_predict_all)):
    best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0.77, 0.87, 0.7]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0.43, 0.67, 0.27]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0.32, 0.50, 0.2]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [0.21, 0.34, 0.13]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [0.77, 0.35, 0.06]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [0, 0.69, 0.94]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [0.75, 0, 0]
        if best_G[i][j] == 8:
            hsi_pic[i, j, :] = [0.7, 0.78, 0.9]
        if best_G[i][j] == 9:
            hsi_pic[i, j, :] = [0.48, 0.48, 0.48]
        if best_G[i][j] == 10:
            hsi_pic[i, j, :] = [0.95, 0.69, 0.51]
        if best_G[i][j] == 11:
            hsi_pic[i, j, :] = [0.97, 0.79, 0.67]
        if best_G[i][j] == 12:
            hsi_pic[i, j, :] = [0.34, 0.34, 0.34]
        if best_G[i][j] == 13:
            hsi_pic[i, j, :] = [0.8, 0.8, 0]
        if best_G[i][j] == 14:
            hsi_pic[i, j, :] = [0, 0.8, 0.4]
        if best_G[i][j] == 15:
            hsi_pic[i, j, :] = [1, 0, 0]

halfwidth = patch_size // 2
utils.classification_map(hsi_pic[halfwidth:-halfwidth, halfwidth:-halfwidth, :], best_G[halfwidth:-halfwidth, halfwidth:-halfwidth], 24,  "classificationMap/HT_{}shot.png".format(TAR_LSAMPLE_NUM_PER_CLASS))









