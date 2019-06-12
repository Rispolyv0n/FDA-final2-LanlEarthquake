def get_bad_feature_name():
    file_path = './data/features_to_be_deleted.txt'
    with open(file_path) as infile:
        content = infile.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x[:x.find(' ')] for x in content]
    return content
