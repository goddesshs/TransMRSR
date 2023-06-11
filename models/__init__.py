from models import MINet, Transmrsr

def create_model(opts):
    opts.model_type = opts.model_type.lower()
    
    #reference-based traning
    if opts.model_type == 'minet':
        model = MINet.RecurrentModel(opts)
   
    else:
        model = Transmrsr.RecurrentModel(opts)

    return model
