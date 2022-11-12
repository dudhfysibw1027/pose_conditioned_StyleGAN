import os
import sys
import torch
import torch.utils.data as data
from parsing_generation_segm_attr_dataset import ParsingGenerationDeepFashionAttrSegmDataset
# from pose_attr_dataset import DeepFashionAttrPoseDataset
from pose_encoder import ShapeAttrEmbedding, PoseEncoder
import numpy as np

class main2:
  def __init__(
    self
    #train_dataset,
    #train_loader,
    
  ):
    self.train_dataset = ParsingGenerationDeepFashionAttrSegmDataset(
        segm_dir='/content/drive/MyDrive/padded_dataset_unzip/padded_dataset/padded_segm',
        pose_dir='/content/drive/MyDrive/padded_dataset_unzip/padded_dataset/padded_densepose',
        ann_file='/content/drive/MyDrive/padded_dataset_unzip/padded_dataset/shape_ann/train_ann_file.txt')

    self.train_loader = data.DataLoader(
        dataset=self.train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
  def main3(self):
  # my_item_dict = train_loader[0]
    for i, data_ in enumerate(self.train_loader):
        if i > 1:
          break
        my_item_dict = data_

    my_pose_encoder = PoseEncoder(size=512)

    my_attr_embedder = ShapeAttrEmbedding(dim=8, out_dim=128,
                                          cls_num_list=[2, 4, 6, 5, 4, 3, 5, 5, 3, 2, 2, 2, 2, 2, 2])

      # print(my_item_dict)
    my_attr_embedding = my_attr_embedder(my_item_dict["attr"])

    x = my_pose_encoder(input=my_item_dict["densepose"], attr_embedding=my_attr_embedding)
    #x = np.array(x)
    return x


# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
#    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
