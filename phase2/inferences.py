import numpy as np
import json
import torch
import torch.utils.data as data
import torch.nn.functional as F

from tqdm import tqdm
from phase2.datasets import FGSBIR_Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(args):
    dataset_test = FGSBIR_Dataset(args, mode='test')
    dataloader_test = data.DataLoader(
        dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=int(args.threads))

    return dataloader_test

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
        mean_rank = torch.zeros(num_steps, dtype=torch.float64, device=device)
        mean_rank_percentile = torch.zeros(num_steps, dtype=torch.float64, device=device)
        
        rank_all = torch.zeros(len(sketch_array_tests), num_steps)
        rank_all_percentile = torch.zeros(len(sketch_array_tests), num_steps)
        
        for i_batch, sampled_batch in enumerate(sketch_array_tests):
            
            sketch_name = sketch_names[i_batch]

            sketch_query_name = '_'.join(
                sketch_name.split('/')[-1].split('_')[:-1])
            position_query = image_names.index(sketch_query_name)
            sketch_features = model.attn(sampled_batch)
            # sketch_features = sampled_batch
            if "CHARUF012YEL-UK_v1_MustardYellow" in sketch_query_name:
                print(sketch_name)
                print(i_batch)
                
            for i_sketch in range(sampled_batch.shape[0]):
                # print("sketch_features[i_sketch].shape: ", sketch_features[i_sketch].shape)
                sketch_feature = sketch_features[i_sketch]
                target_distance = F.pairwise_distance(sketch_feature.to(device), image_array_tests[position_query].to(device))
                distance = F.pairwise_distance(sketch_feature.unsqueeze(0).to(device), image_array_tests.to(device))
                
                if sketch_name == "/test/CHARUF012YEL-UK_v1_MustardYellow_12" and i_sketch == 2:
                    sorted_dist, sorted_idx = torch.sort(distance)
                    top_idx = sorted_idx[:10].tolist()
                    top_names = [image_names[i] for i in top_idx]
                    
                    print(top_names)
                
                rank_all[i_batch, i_sketch] = distance.le(target_distance).sum()
                rank_all_percentile[i_batch, i_sketch] = (len(distance) - rank_all[i_batch, i_sketch]) / (len(distance) - 1)
                
            
        print(rank_all[22, :].numpy())
    
def inference_model(model, args):
    model = model.to(device)
    model.load_state_dict(torch.load(args.pretrained_dir), strict=False)
    dataloader_test = get_dataloader(args)
    for i_epoch in range(args.epochs):
        print(f"Epoch: {i_epoch+1} / {args.epochs}")
        
        avererage_area, avererage_area_percentile = evaluate_model(model=model, dataloader_test=dataloader_test)
        with open("results.json", "w") as f:
            json.dump({
                "avererage_area": avererage_area,
                "avererage_area_percentile": avererage_area_percentile,
            }, f)