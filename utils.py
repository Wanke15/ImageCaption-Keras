def load_flicker8k_token(file_name):
    # read text file
    with open(file_name, 'r') as f:
        lines = f.read()
    img2desc = dict()
    # process lines
    for line in lines.split('\n'):
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # extract filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        # create the list if needed
        if image_id not in mapping:
            img2desc[image_id] = list()
        # store description
        img2desc[image_id].append(image_desc)
    return img2desc


