import os

from core.config import DATASET_PATH


DATASET = os.path.join(DATASET_PATH, 'voxceleb1')


def get_metadata():
    metadata_path = os.path.join(DATASET, 'vox1_meta.csv')
    with open(metadata_path, mode="r", encoding="utf-8") as file:
        data = [line.replace('\n', '').split('\t')
                for line in file.readlines()]
        headers = data[0]
        data = data[1:]

        vox_dict = {}
        for celeb_info in data:
            # Create a dictionary that maps celeb_id to
            # each celeb properties dictionary
            vox_celeb_id = celeb_info[0]
            vox_dict[vox_celeb_id] = {k: v for k,
                                      v in zip(headers[1:], celeb_info[1:])}
        return vox_dict


print(get_metadata())
