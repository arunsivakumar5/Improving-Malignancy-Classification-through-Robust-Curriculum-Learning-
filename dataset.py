


class LIDC_Dataset(Dataset):
    def __init__(self,features,labels,subclasses):
        '''
        INPUTS:
        features: list of features (as Pytorch tensors)
        labels:   list of corresponding lables
        subclasses: list of corresponding subclasses
        '''

        self.X = features
        self.features = features
        self.labels = labels
        self.subclasses = subclasses

    
        yList = []
 
        for i in labels:
            if i == 0:
                yList.append(0)
            elif i ==1:
                yList.append(1)
            else:
                yList.append(2)
        subYList = []
        print("subclasses",subclasses)
        for j in subclasses:
            print(j)
            if j == 0:
                subYList.append(0)
            elif j ==1:
                subYList.append(1)
            elif j == 2:
                subYList.append(2)
            elif j == 3:
                subYList.append(3)
            elif j == 4:
                subYList.append(4)
                
        self.Y_dict = {}
        self.Y_dict['superclass'] = torch.tensor(yList)
        self.Y_dict['subclass'] = torch.tensor(subYList)
                
                
                
    def __getitem__(self, index):
        
        return self.X[index], {'superclass':self.Y_dict['superclass'][index], 'subclass':self.Y_dict['subclass'][index]}
        
    
    def __len__(self):
        return len(self.X)
    
    def get_class_counts(self, className):
        return torch.tensor(np.bincount(np.array(self.Y_dict[className])))