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