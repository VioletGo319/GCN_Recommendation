import pandas as pd 
import numpy as np




if __name__ == "__main__":


    user_list = pd.read_csv('https://raw.githubusercontent.com/kuandeng/LightGCN/refs/heads/master/Data/amazon-book/user_list.txt', dtype={"org_id": str, "remap_id": int},
    sep = '\s+')
    user_list.rename(columns = {"org_id":"user_id"}, inplace = True)

    
    train_list = pd.read_csv('https://raw.githubusercontent.com/kuandeng/LightGCN/refs/heads/master/Data/amazon-book/train.txt')
    test_list = pd.read_csv('https://raw.githubusercontent.com/kuandeng/LightGCN/refs/heads/master/Data/amazon-book/test.txt')

    
    item_list = pd.read_csv('https://raw.githubusercontent.com/kuandeng/LightGCN/master/Data/amazon-book/item_list.txt', sep=r"\s+", engine="python",
                        dtype={"org_id": str, "remap_id": int})
    item_list.rename(columns = {"org_id":"item_id"}, inplace = True)








