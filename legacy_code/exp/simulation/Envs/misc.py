import pandas as pd
from termcolor import cprint 

rprint = lambda text: cprint(text, 'red')

def feature_gen(ls):
    def feature(method):
        ls.append(method)
        return method
    return feature

class InstanceExistingError(Exception):
    pass

class FeaturesBase:
    _instance_exist = 0
    feature_list = []
    feature = feature_gen(feature_list)
    def __init__(self):
        if FeaturesBase._instance_exist:
            raise InstanceExistingError('Number of instance should be no more than 1.')
        FeaturesBase._instance_exist += 1 

feature = FeaturesBase.feature

class Recorder:
    def __init__(self, max_size, fields, overwrite = False):
        self.max_size = max_size
        self.columns = fields
        self.num_col = len(fields)
        self.data = pd.DataFrame(columns = fields)
        self.idx = 0
        self.size = 0
        self.is_overwrite = overwrite
        self._dump_flag = False

    def record(self, *args, **kwargs):
        assert len(args)==0, f'this method only takes keyword arguments.'
        if self.max_size is not None:
            assert self.size <= self.max_size, f'maximal size is reached: {self.size}/{self.max_size}'
        col = list(kwargs.keys())
        
        assert len([item for item in col if item not in self.columns]) == 0, f'check keys: {kwargs.keys()}'
        assert len(kwargs)==self.num_col, f'check the number of keys: {len(kwargs)}/{self.num_col}'
        self.data.loc[self.idx, :] = [kwargs[key] for key in kwargs.keys()]
        self.idx += 1
        self.size += 1
        self._dump_flag = False
    
    @property
    def table(self):
        return self.data

    def dump(self, file_, *arg, **kwargs):
        self._dump_flag = True
        return self.data.to_csv(file_, *arg, **kwargs)

    def reset(self):
        if not self._dump_flag:
            rprint('Recorder Warning: erasing unsaved data.')
        self.data = pd.DataFrame(columns = self.columns)
        self.idx = 0
        self.size = 0

    


if __name__ == "__main__":

    # ---------- Record test -----------
    import os
    os.chdir('dev/Env')
    print(os.getcwd())

    rec = Recorder(30, ['a', 'b', 'c'], overwrite=True)
    rec.record(a = 1,b = 2, c = 3)
    # rec.record(a = 1, b = 2)
    # rec.record(1,2,3)
    # rec.record(a = 1, b = 2, d = 3)
    for i in range(20):
        rec.record(a = i*1, b = i*10, c = i*100)
    print(rec.table)
    rec.dump('test.csv')
    # ------------ Record test ends -------------


    



