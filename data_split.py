import os
import math
import json
import numpy as np
import pandas  as pd

import collections
import torch

def data_split(rootDir, filename):
              
        select_ID=pd.read_csv(filename)
        select_ID.drop(select_ID[select_ID['FILE_ID']=='no_filename'].index, inplace=True)
        select_ID.drop(select_ID[select_ID['func_mean_fd']>=0.2].index, inplace=True)
          
        data = os.listdir(rootDir)
        
        data_set = collections.defaultdict(list)
        annot_data_set = collections.defaultdict(list)

        
        list_data = []; data_path = []; data_path_full = []
        for i in data:
            folder_list = []; path_list = []; path_list_full = []
            for j in os.listdir(os.path.join(rootDir, i)):
                for k in os.listdir(os.path.join(rootDir, i, j)):
                    if k.split('.')[-1] == 'gz':
                        path_list.append(k.split('func_preproc')[0])
                        path_list_full.append(os.path.join(rootDir,i,j,k))
                list_data.append(folder_list)
                data_path.append(path_list)
                data_path_full.append(path_list_full)
        index = []; ASD_annotations = []; TC_annotations = []; genre = []; ASD = []; TC = []
       
        for j,i in enumerate(data_path[2]):
            for l,k in enumerate(select_ID['FILE_ID'].values):
                if i == (k + str('_')):
                    if select_ID['DX_GROUP'].values[l]==1:
                        ASD_annotations.append(0)  #ASD
                        ASD.append(j)
                    if select_ID['DX_GROUP'].values[l]==2:
                        TC_annotations.append(1)  #TC
                        TC.append(j)
                    index.append(j)
                    if select_ID['SEX'].values[l]==1:
                        genre.append(0)    #ASD
                    if select_ID['SEX'].values[l]==2:
                        genre.append(1)    #TC

        #ASD_train = math.ceil(0.7 * len(ASD))
        ASD_train = len(ASD[:3])
        #ASD_valid = math.ceil(0.2 * len(ASD))
        ASD_valid = len(ASD[3:6])
        #ASD_test = len(ASD) - ASD_train - ASD_valid
        ASD_test = len(ASD[6:9])
        
        ASDtrain_split, ASDtest_split, ASDvalid_split = torch.utils.data.random_split(ASD[:9], (ASD_train, ASD_test, ASD_valid))

        #ASD_train_annotations = math.ceil(0.7 * len(ASD_annotations))
        ASD_train_annotations = len(ASD_annotations[:3])
        #ASD_valid_annotations = math.ceil(0.2 * len(ASD_annotations))
        ASD_valid_annotations = len(ASD_annotations[3:6])
        #ASD_test_annotations = len(ASD_annotations) - ASD_train_annotations - ASD_valid_annotations
        ASD_test_annotations = len(ASD_annotations[6:9])
        ASDtrain_annot_split, ASDtest_annot_split, ASDvalid_annot_split = torch.utils.data.random_split(ASD_annotations[:9], (ASD_train_annotations, ASD_test_annotations, ASD_valid_annotations))
        
        #TC_train = int(0.7 * len(TC))
        TC_train = len(TC[:3])
        #TC_valid = int(0.2 * len(TC))
        TC_valid = len(TC[3:6])
        #TC_test = len(TC) - TC_train - TC_valid
        TC_test = len(TC[6:9])
        TCtrain_split, TCtest_split, TCvalid_split = torch.utils.data.random_split(TC[:9], (TC_train, TC_test, TC_valid))

        #TC_train_annotations = int(0.7 * len(TC_annotations))
        TC_train_annotations = len(TC_annotations[:3])
        #TC_valid_annotations = int(0.2 * len(TC_annotations))
        TC_valid_annotations = len(TC_annotations[3:6])
        #TC_test_annotations = len(TC_annotations) - TC_train_annotations - TC_valid_annotations
        TC_test_annotations = len(TC_annotations[6:9])
        TCtrain_annot_split, TCtest_annot_split, TCvalid_annot_split = torch.utils.data.random_split(TC_annotations[:9], (TC_train_annotations, TC_test_annotations, TC_valid_annotations))

        
        for i in ASDtrain_split:
                data_set['train'].append({ 
                'img':data_path_full[0][i]
                })
        for i in ASDtrain_annot_split:
                annot_data_set['train'].append({ 
                'annot':i
                })
        for i in TCtrain_split:
                data_set['train'].append({ 
                'img':data_path_full[0][i]
                })
        for i in TCtrain_annot_split:
                annot_data_set['train'].append({ 
                'annot':i
                })
                
                
        for i in ASDtest_split:
                data_set['test'].append({ 
                'img':data_path_full[0][i]
                })
        for i in ASDtest_annot_split:
                annot_data_set['test'].append({ 
                'annot':i
                })
        for i in TCtest_split:
                data_set['test'].append({ 
                'img':data_path_full[0][i]
                })
        for i in TCtest_annot_split:
                annot_data_set['test'].append({ 
                'annot':i
                })
                
                
        for i in ASDvalid_split:
                data_set['valid'].append({ 
                'img':data_path_full[0][i]
                })
        for i in ASDvalid_annot_split:
                annot_data_set['valid'].append({ 
                'annot':i
                })
        for i in TCvalid_split:
                data_set['valid'].append({ 
                'img':data_path_full[0][i]
                })
        for i in TCvalid_annot_split:
                annot_data_set['valid'].append({ 
                'annot':i
                })

        with open('data_example_6.json', 'w') as fp:
                json.dump(data_set, fp)

        with open('annotation_example_6.json', 'w') as fp:
                json.dump(annot_data_set, fp)
        #breakpoint()
        return data_set, annot_data_set

rootDir = os.path.join('/media/disk1/user_home1/lvbellon/PROJECT/Outputs/cpac')
filename = r'/home/lvbellon/PROJECT/Phenotypic_V1_0b_preprocessed1.csv'

data_split = data_split(rootDir, filename)
#breakpoint()
#print(data_split)