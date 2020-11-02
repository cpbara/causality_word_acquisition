def action_recognition_splits(root_path, train_file, test_file):
    files = {}
    fun = lambda x: '/'.join([root_path, x]).strip().split(' ')
    with open(train_file) as f:
        files['training'] = [fun(x) for x in f.readlines()]
    with open(test_file) as f:
        testvalset = [fun(x) for x in f.readlines()]
        files['testing']    = testvalset[0::2]
        files['validation'] = testvalset[1::2]
    return files