from MoRec import MoRec
from util.conf import ModelConf
import torch

if __name__ == '__main__':
    # Register your model here
    

    # torch.cuda.set_device(1)
    # current_device = torch.cuda.current_device()
    
    baselines= ['SASRec','BERT4Rec','CL4SRec','UnisRec']
    #可以进config
    stages=['Pre','Fine']
    
    print('=' * 80)
    print('   MoRec: A library for Modal Recommendation.   ')
    print('=' * 80)

    print('Baseline Models:')
    print('   '.join(baselines))
    print('-' * 80)
    model = input('Please enter the baselines you want to run:')
    
    #model = 'BERT4Rec'
    print('stages:')
    print('   '.join(stages))
    print('-' * 80)
    stage=input('Please enter the stage you want:')
    import time

    s = time.time()
    if model in baselines:
        conf = ModelConf('./conf/' + model + '.conf',stage)
    else:
        print('Wrong model name!')
        exit(-1)

    rec = MoRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
