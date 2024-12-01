# @ Haotian Xue 2023
# accelerated version: mist v3
# feature 1: SDS
# feature 2: Diff-PGD

import os
import ssl
import glob
import torch

import PIL
from PIL import Image
from tqdm import tqdm
from ldm.util import instantiate_from_config

from utils import cprint, load_png, mp
from utils import lpips_, ssim_, psnr_

from utils import lpips_, ssim_, psnr_

from clip_similarity import calculate_clip_score


ssl._create_default_https_context = ssl._create_unverified_context
os.environ['TORCH_HOME'] = os.getcwd()
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hub/')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_image_from_path(image_path: str, input_size: int) -> PIL.Image.Image:
    """
    Load image form the path and reshape in the input size.
    :param image_path: Path of the input image
    :param input_size: The requested size in int.
    :returns: An :py:class:`~PIL.Image.Image` object.
    """
    img = Image.open(image_path).resize((input_size, input_size),
                                        resample=PIL.Image.BICUBIC)
    return img


def load_model_from_config(config, ckpt, verbose: bool = False):
    """
    Load model from the config and the ckpt path.
    :param config: Path of the config of the SDM model.
    :param ckpt: Path of the weight of the SDM model
    :param verbose: Whether to show the unused parameters weight.
    :returns: A SDM model.
    """
    print(f"Loading model from {ckpt}")

    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    # Support loading weight from NovelAI
    if "state_dict" in sd:
        import copy
        sd_copy = copy.deepcopy(sd)
        for key in sd.keys():
            if key.startswith('cond_stage_model.transformer') and not key.startswith('cond_stage_model.transformer.text_model'):
                newkey = key.replace('cond_stage_model.transformer', 'cond_stage_model.transformer.text_model', 1)
                sd_copy[newkey] = sd[key]
                del sd_copy[key]
        sd = sd_copy

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


def get_dir_name_from_config(mode, g_mode, using_target, eps=16, steps=100, target_rate=5, prefix='output_target'):
    
    if mode == 'none':
        return f'{prefix}/none/'
    
    if using_target and mode == 'sds':
        mode_name = f'sdsT{target_rate}'
    else:
        mode_name = mode
    dir_name = f'{prefix}/{mode_name}_eps{eps}_steps{steps}_gmode{g_mode}/'
    return dir_name


# EXP_LIST = [
#     ('advdm', '+', False, -1),
#     ('advdm', '-', False, -1),
#     ('mist', '+', False, -1),
#     ('sds', '+', False, -1),
#     ('sds', '-', False, -1),
#     ('sds', '-', True, 5),
#     ('texture_only', '+', False, -1),
#     ('none', '-', False, -1)
# ]
EXP_LIST = [
    ('mist', '+', False, -1),
    ('mist', '-', False, -1),
    ('texture_only', '+', False, -1),
    ('texture_only', '-', False, -1),
    ('none', '-', False, -1)
]


@torch.no_grad()
def main():
    
    for exp_config in tqdm(EXP_LIST):
        cprint(exp_config, 'y')
        mode, g_mode, using_target, target_rate = exp_config
        
        adv_dir = get_dir_name_from_config(mode, g_mode, using_target, target_rate=target_rate)
        cprint('fetching dir: ' + adv_dir, 'g')
        
        clean_dir = get_dir_name_from_config('none', '-', using_target=False, target_rate=target_rate)
        save_path = get_dir_name_from_config(mode, g_mode, using_target, target_rate=target_rate, prefix='eval')
        mp(save_path)
            
        clip_score_list = []

        ssim_list = []
        psnr_list = []
            
        x_list = []
        x_adv_list =[]
        
        adv_img_paths = glob.glob(adv_dir + '/*_inpaint.png') 
        adv_img_paths.sort(key=lambda x: int(x[x.rfind('/') + 1 : x.rfind('_')]))

        clean_img_paths = glob.glob(clean_dir + '/*_inpaint.png') 
        clean_img_paths.sort(key=lambda x: int(x[x.rfind('/') + 1 : x.rfind('_')]))

        for i in tqdm(range(len(adv_img_paths))):
            adv_img_path = adv_img_paths[i]
            clean_img_path = clean_img_paths[i]
            
            if not os.path.exists(adv_img_path):
                print("NO SUCH PATH", adv_img_path)
                break
        
            x_adv = load_png(adv_img_path, 512)[None, ...].to(device)
            x = load_png(clean_img_path, 512)[None, ...].to(device)
             
            
            x_list.append(x)
            x_adv_list.append(x_adv)

            ssim_x = ssim_(adv_img_path, clean_img_path)
            psnr_x = psnr_(adv_img_path, clean_img_path)
            ssim_list.append(ssim_x)
            psnr_list.append(psnr_x)
            
            clip_score = calculate_clip_score(adv_img_path, 'a person in a restaurant')
            clip_score_list.append(clip_score)

        x_adv_all = torch.cat(x_adv_list, 0)
        x_all = torch.cat(x_list, 0)
        lpips_score = lpips_(x_all, x_adv_all)
        lpips_score = lpips_score[:, 0, 0, 0].cpu().tolist()

        torch.save({
            'ssim':ssim_list,
            'lpips':lpips_score,
            'psnr':psnr_list,
            'clip_score': clip_score_list
        }, save_path +'/inpaint_metrics.bin')

        cprint({
            'ssim':ssim_list,
            'lpips':lpips_score,
            'psnr':psnr_list,
            'clip_score': clip_score_list
        }, 'y')
            

if __name__ == '__main__':
    main()
            
                
                
                
                
                
                
                
                
                
                
                
                
            

    # from utils import load_png
    # x = load_png(x_path, 512)[None, ...].to(device)
    # x_adv = load_png(x_adv_path, 512)[None, ...].to(device)

    # x = x * 2 - 1
    # x_adv = x_adv * 2 - 1

    # print(torch.abs(x-x_adv).max() * 255)
    # print("l1", torch.abs(x-x_adv).norm(p=1))

    # z = model.get_first_stage_encoding(model.encode_first_stage(x)).to(device)
    # z_adv = model.get_first_stage_encoding(model.encode_first_stage(x_adv)).to(device)
    

    
    # x_decode = model.decode_first_stage(z, force_not_quantize=True)
    # x_adv_decode = model.decode_first_stage(z_adv, force_not_quantize=True)
    
    

    
    

if __name__ == '__main__':
    main()