import os
import json

# fix random seed
def fixed_seed(seed):
    import numpy as np
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # make some cudnn algorythm determiniscic
        torch.backends.cudnn.deterministic = True
        # NOTE: if it False process might be slow
        torch.backends.cudnn.benchmark = True

# handles pytorch x numpy seeding issue
def worker_init_fn(worker_id):
    import numpy as np
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def _support_function_type_default(o):
    import types
    if isinstance(o, (types.FunctionType, type)):
        return o.__name__
    raise TypeError('{} is not JSON serializable'.format(repr(o)))

# save dict as JSON
def save_json(dic, filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    
    with open(filename, 'w') as f:
        json.dump(dic, f, indent = 4, default=_support_function_type_default)

# load JSON as dict
def load_json(filename):
    _, ext = os.path.splitext(os.path.basename(filename))

    if ext != '.json':
        print('{} is not json file'.format(filename))
        raise NotImplementedError()

    with open(filename) as f:
        j = json.load(f)
    
    return j