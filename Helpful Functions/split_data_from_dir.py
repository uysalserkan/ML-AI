def split_data(source, training, testing, split_size=0.8, shuffle=1):
    files = []
    # Getting files from direction
    for filename in os.listdir(source):
        file = source + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so this file ignoring..")
    
    train_len = int(len(files) * split_size)
    test_len = int(len(files) - train_len)

    if shuffle:
        shuffled_set = random.sample(files, len(files))
        training_set = shuffled_set[0:train_len]
        testing_set = shuffled_set[train_len:]
    else:
        training_set = files[0:train_len]
        testing_set = files[train_len:]

    for filename in training_set:
        this_file = source + filename
        dest = training + filename
        copyfile(this_file, dest)

    for filename in testing_set:
        this_file = source + filename
        dest = testing + filename
        copyfile(this_file, dest)

