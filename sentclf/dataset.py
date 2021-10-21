from torch.utils.data import Dataset


class DataFrameDataset(Dataset):
    def __init__(self, dataframe):
        super().__init__()
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        return self.dataframe.iloc[index].to_dict()


class ScientificPaperDataset(DataFrameDataset):
    def __init__(self, dataframe):
        super().__init__(dataframe)
        self.__transform()

    def __transform(self):
        self.dataframe['label'] = self.dataframe['label'].cat.codes.astype('int')

    @property
    def sentences(self):
        return self.dataframe['sentence']

    @property
    def labels(self):
        return self.dataframe['label']
