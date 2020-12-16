from glob import glob
import os

def action_recognition_splits_2(root_path, train_file, test_file, train_size = None):
    files = {}
    class_dict = dict([(x,i+1) for i,x in enumerate(sorted(glob(os.path.join(root_path,'*'))))])
    paths = sorted(glob(os.path.join(root_path,'*/*/*/*')))
    train_paths = sum([paths[i::10] for i in range(7)],[])
    testavl_paths = sum([paths[i::10] for i in range(7,10)],[])
    if train_size is None:
        files['training'] = [(path,class_dict[path.rsplit('/',3)[0]]) for path in train_paths]
    else:
        files['training'] = []
        for i in range(min(train_size,70)):
            files['training'] += [(path,class_dict[path.rsplit('/',3)[0]]) for path in train_paths[i::70]]
    files['validation'] = [(path,class_dict[path.rsplit('/',3)[0]]) for path in testavl_paths[0::2]]
    files['testing'] = [(path,class_dict[path.rsplit('/',3)[0]]) for path in testavl_paths[1::2]]
    return files

def action_recognition_splits(root_path, train_file, test_file, train_size = None):
    files = {}
    fun = lambda x: '/'.join([root_path, x]).strip().split(' ')
    with open(train_file) as f:
        if train_size is None:
            files['training'] = [fun(x) for x in f.readlines()]
        else:
            fls = [fun(x) for x in f.readlines()]
            files['training'] = []
            for i in range(min(train_size,70)):
                files['training'] += fls[i::70]
    with open(test_file) as f:
        testvalset = [fun(x) for x in f.readlines()]
        files['testing']    = testvalset[0::2]
        files['validation'] = testvalset[1::2]
    return files