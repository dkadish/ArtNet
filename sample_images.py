#!python3
#
# Download a sample of images

import hashlib
import pandas as pd
import time
import requests
from tqdm import tqdm

DATAFILE = 'smk_all_artworks.json'
N = 1000
SEED = 2019                    # Because it is as good a number as any
DATA_DIR = 'data/images/sample1000'

assert hashlib.md5(open(DATAFILE,'rb').read()).hexdigest() == '83edafa71f53e51f45258c248b928e44'
# FIXME: that data dir exists

smk = pd.read_json(DATAFILE).set_index('id')
print(smk.shape)

# Whoops using "pd" both for Pandas as Public Domain
smk_pd = smk[smk['public_domain'] & smk['has_image']]

sample = smk_pd.sample(N, random_state=SEED)

# assert smk['id'].is_unique

for i, data in tqdm(sample[smk['has_image']].iterrows()):
    with open(f"{DATA_DIR}/{i}.jpg", 'wb') as fd:
        resp = requests.get(data['image_native'])
        fd.write(resp.content)
        time.sleep(2)
