import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from tqdm import tqdm
from torch import optim
from phase2.datasets import FGSBIR_Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(args):
    dataset_train = FGSBIR_Dataset(args, mode='train')
    dataloader_train = data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.threads))

    dataset_test = FGSBIR_Dataset(args, mode='test')
    dataloader_test = data.DataLoader(
        dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=int(args.threads))

    return dataloader_train, dataloader_test

def evaluate_model(model, dataloader_test):
    with torch.no_grad():
        model.eval()
        sketch_array_tests = []
        sketch_names = []
        image_array_tests = torch.FloatTensor().to(device)
        image_names = []

        for idx, batch in enumerate(tqdm(dataloader_test)):
            sketch_features_all = torch.FloatTensor().to(device)
            # print(batch['sketch_imgs'].shape) # (1, 25, 3, 299, 299)
            
            for data_sketch in batch['sketch_imgs']:
                sketch_feature = model.sketch_embedding_network(
                    data_sketch.to(device))
                sketch_feature = model.sketch_attention(sketch_feature)
                # sketch_feature = model.sketch_linear(sketch_feature)
                
                # print("sketch_feature.shape: ", sketch_feature.shape) #(25, 2048)
                sketch_features_all = torch.cat(
                    (sketch_features_all, sketch_feature.detach()))

            # print("sketch_feature_ALL.shape: ", sketch_features_all.shape) # (25, 2048)
            sketch_array_tests.append(sketch_features_all)
            sketch_names.extend(batch['sketch_path'])

            if batch['positive_path'][0] not in image_names:
                positive_feature = model.sample_embedding_network(
                    batch['positive_img'].to(device))
                positive_feature = model.linear(
                    model.attention(positive_feature))
                # positive_feature, _ = model.attention(
                #     model.sample_embedding_network(batch['positive_img'].to(device)))
                image_array_tests = torch.cat(
                    (image_array_tests, positive_feature))
                image_names.extend(batch['positive_path'])

        # print("sketch_array_tests[0].shape", sketch_array_tests[0].shape) #(25, 2048)
        num_steps = len(sketch_array_tests[0])
        avererage_area = []
        avererage_area_percentile = []
        mean_rank_ourB = []
        mean_rank_ourA = []
        avererage_ourB = []
        avererage_ourA = []
        exps = np.linspace(1, num_steps, num_steps) / num_steps
        factor = np.exp(1 - exps) / np.e
        sketch_range = []
        
        rank_all = torch.zeros(len(sketch_array_tests), num_steps)
        rank_all_percentile = torch.zeros(len(sketch_array_tests), num_steps)
        sketch_range = torch.Tensor(sketch_range)
        
        for i_batch, sampled_batch in enumerate(sketch_array_tests):
            mean_rank = []
            mean_rank_percentile = []
            sketch_name = sketch_names[i_batch]

            sketch_query_name = '_'.join(
                sketch_name.split('/')[-1].split('_')[:-1])
            position_query = image_names.index(sketch_query_name)
            sketch_features = model.attn(sampled_batch)
            # sketch_features = sampled_batch

            for i_sketch in range(sampled_batch.shape[0]):
                # print("sketch_features[i_sketch].shape: ", sketch_features[i_sketch].shape)
                sketch_feature = sketch_features[i_sketch]
                target_distance = F.pairwise_distance(sketch_feature.to(device), image_array_tests[position_query].to(device))
                distance = F.pairwise_distance(sketch_feature.unsqueeze(0).to(device), image_array_tests.to(device))
                
                rank_all[i_batch, i_sketch] = distance.le(target_distance).sum()
                rank_all_percentile[i_batch, i_sketch] = (len(distance) - rank_all[i_batch, i_sketch]) / (len(distance) - 1)
                
                if rank_all[i_batch, i_sketch].item() == 0:
                    mean_rank.append(1.)
                else:
                    mean_rank.append(1/rank_all[i_batch, i_sketch].item())
                    # 1/(rank)
                    mean_rank_percentile.append(rank_all_percentile[i_batch, i_sketch].item())
                    mean_rank_ourB.append(1/rank_all[i_batch, i_sketch].item() * factor[i_sketch])
                    mean_rank_ourA.append(rank_all_percentile[i_batch, i_sketch].item()*factor[i_sketch])
                    
            avererage_area.append(np.sum(mean_rank)/len(mean_rank))
            avererage_area_percentile.append(np.sum(mean_rank_percentile)/len(mean_rank_percentile))
            avererage_ourB.append(np.sum(mean_rank_ourB)/len(mean_rank_ourB))
            avererage_ourA.append(np.sum(mean_rank_ourA)/len(mean_rank_ourA))

        top1_accuracy = rank_all[:, -1].le(1).sum().numpy() / rank_all.shape[0]
        top5_accuracy = rank_all[:, -1].le(5).sum().numpy() / rank_all.shape[0]
        top10_accuracy = rank_all[:, -1].le(10).sum().numpy() / rank_all.shape[0]

        meanMA = np.mean(avererage_area_percentile)
        meanMB = np.mean(avererage_area)
        meanOurB = np.mean(avererage_ourB)
        meanOurA = np.mean(avererage_ourA)

        return top1_accuracy, top5_accuracy, top10_accuracy, meanMA, meanMB, meanOurA, meanOurB

def train_model(model, args):
    model = model.to(device)
    dataloader_train, dataloader_test = get_dataloader(args)
    if args.load_pretrained:
        model.load_state_dict(torch.load(args.pretrained_dir), strict=False)

    loss_fn = nn.TripletMarginLoss(margin=args.margin)
    # optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    optimizer = optim.AdamW([
        # {'params': model.sketch_linear.parameters(), 'lr': args.lr},
        {'params': model.attn.parameters(), 'lr': args.lr},
    ])
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    top5, top10, avg_loss = 0, 0, 0
    for i_epoch in range(args.epochs):
        print(f"Epoch: {i_epoch+1} / {args.epochs}")

        losses = []
        for _, batch_data in enumerate(tqdm(dataloader_train, dynamic_ncols=False)):
            model.sketch_embedding_network.eval()
            model.sample_embedding_network.eval()
            model.sketch_attention.eval()
            model.attention.eval()
            model.linear.eval()
            model.attn.train()
            # model.sketch_linear.train()
            optimizer.zero_grad()
            positive_features = model.sample_embedding_network(batch_data['positive_img'].to(device))
            negative_features = model.sample_embedding_network(batch_data['negative_img'].to(device))
            positive_features = model.linear(model.attention(positive_features))
            negative_features = model.linear(model.attention(negative_features))
            
            loss = 0
            
            for idx in range(len(batch_data['sketch_imgs'])): # len(batch_data['sketch_imgs']) = batch_size
                sketch_seq_feature = model.sketch_embedding_network(batch_data['sketch_imgs'][idx].to(device))
                sketch_seq_feature = model.attn(model.sketch_attention(sketch_seq_feature))
                
                positive_feature = positive_features[idx]
                negative_feature = negative_features[idx]
                
                positive_feature = positive_feature.repeat(sketch_seq_feature.shape[0], 1)
                negative_feature = negative_feature.repeat(sketch_seq_feature.shape[0], 1)
                
                loss += loss_fn(sketch_seq_feature, positive_feature, negative_feature)      
            
            loss.backward()
            optimizer.step()
            # scheduler.step()

            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)

                
        top1_eval, top5_eval, top10_eval, meanA, meanB, meanOurA, meanOurB = evaluate_model(model=model, dataloader_test=dataloader_test)
        if top5_eval > top5:
            top5 = top5_eval
            torch.save(model.state_dict(), "best_top5_model.pth")
        if top10_eval > top10:
            top10 = top10_eval
            torch.save(model.state_dict(), "best_top10_model.pth")
            
        torch.save(model.state_dict(), "last_model.pth")
        print('Top 1 accuracy : {:.5f}'.format(top1_eval))
        print('Top 5 accuracy : {:.5f}'.format(top5_eval))
        print('Top 10 accuracy: {:.5f}'.format(top10_eval))
        print('Mean A         : {:.5f}'.format(meanA))
        print('Mean B         : {:.5f}'.format(meanB))
        print('meanOurA       : {:.5f}'.format(meanOurA))
        print('meanOurB       : {:.5f}'.format(meanOurB))
        print('Loss           : {:.5f}'.format(avg_loss))
        with open("results_log.txt", "a") as f:
            f.write("Epoch {:d} | Top1: {:.5f} | Top5: {:.5f} | Top10: {:.5f} | MeanA: {:.5f} | MeanB: {:.5f} | meanOurA: {:.5f} | meanOurB: {:.5f} \n".format(
                i_epoch+1, top1_eval, top5_eval, top10_eval, meanA, meanB, meanOurA, meanOurB))