import torch
from thop import profile


def complexity(model):

    data = {'ref_image_full': torch.randn(1,3,256,256),
                    'ref_image_sub': torch.randn(1,3,256,256),
                    # 'ref_kspace_full': T1_ks,

                    'tag_image_full': torch.randn(1,3,256,256),

                    'tag_image_sub': torch.randn(1,3,256,256),
                    # 'tag_image_sub_sub': T2_128_img,
                    # 'tag_kspace_sub': T2_128_ks,
                    }
    model.set_input(data)
    flops, params = profile(model, data['ref_image_full'])
    print('flops: ', flops, 'params: ', params)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters: {} \n'.format(num_param))

def tensor_transform_reverse(image):
    assert image.dim() == 4
    moco_input = torch.zeros(image.size()).type_as(image)
    moco_input[:,0,:,:] = image[:,0,:,:] * 0.229 + 0.485
    moco_input[:,1,:,:] = image[:,1,:,:] * 0.224 + 0.456
    moco_input[:,2,:,:] = image[:,2,:,:] * 0.225 + 0.406
    return moco_input


