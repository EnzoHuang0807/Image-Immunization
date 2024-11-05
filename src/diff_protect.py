# @ Haotian Xue 2023
# accelerated version: mist v3
# feature 1: SDS
# feature 2: Diff-PGD

import os
import numpy as np
from omegaconf import DictConfig, OmegaConf
import PIL
from PIL import Image, ImageOps
from einops import rearrange
import ssl
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config
from advertorch.attacks import LinfPGDAttack
from attacks import Linf_PGD, SDEdit
from diffusers import StableDiffusionInpaintPipeline
import time
import wandb
import glob
import hydra
from utils import *



ssl._create_default_https_context = ssl._create_unverified_context
os.environ['TORCH_HOME'] = os.getcwd()
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hub/')
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

totensor = transforms.ToTensor()
topil = transforms.ToPILImage()


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


def load_model_from_config(config, ckpt, verbose: bool = False, device=0):
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
    model.cond_stage_model.to(device)
    model.eval()
    return model


class identity_loss(nn.Module):
    """
    An identity loss used for input fn for advertorch. To support semantic loss,
    the computation of the loss is implemented in class targe_model.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x


class target_model(nn.Module):
    """
    A virtual model which computes the semantic and textural loss in forward function.
    """

    def __init__(self, model,
                 condition: str,
                 target_info: str = None,
                 mode: int = 2, 
                 rate: int = 10000, g_mode='+', device=0):
        """
        :param model: A SDM model.
        :param condition: The condition for computing the semantic loss.
        :param target_info: The target textural for textural loss.
        :param mode: The mode for computation of the loss. 0: semantic; 1: textural; 2: fused
        :param rate: The fusion weight. Higher rate refers to more emphasis on semantic loss.
        """
        super().__init__()
        self.model = model
        self.condition = condition
        self.fn = nn.MSELoss(reduction="sum")
        self.target_info = target_info
        self.mode = mode
        self.rate = rate
        self.g_mode = g_mode
        
        print('g_mode:',  g_mode)

    def get_components(self, x):
        """
        Compute the semantic loss and the encoded information of the input.
        :return: encoded info of x, semantic loss
        """

        z = self.model.get_first_stage_encoding(self.model.encode_first_stage(x)).to(x.device)
        c = self.model.get_learned_conditioning(self.condition)
        loss = self.model(z, c)[0]
        return z, loss

    def forward(self, x, components=False):
        """
        Compute the loss based on different mode.
        The textural loss shows the distance between the input image and target image in latent space.
        The semantic loss describles the semantic content of the image.
        :return: The loss used for updating gradient in the adversarial attack.
        """
        g_dir = 1. if self.g_mode == '+' else -1.

        
        zx, loss_semantic = self.get_components(x)
        zy, loss_semantic_y = self.get_components(self.target_info)
        
        if components:
            return self.fn(zx, zy), loss_semantic
        if self.mode == 'advdm': # 
            return - loss_semantic * g_dir
        elif self.mode == 'texture_only':
            return self.fn(zx, zy)
        elif self.mode == 'mist':
            return self.fn(zx, zy) * g_dir  - loss_semantic * self.rate
        elif self.mode == 'texture_self_enhance':
            return - self.fn(zx, zy)
        else:
            raise KeyError('mode not defined')


def init(epsilon: int = 16, steps: int = 100, alpha: int = 1, 
         input_size: int = 512, object: bool = False, seed: int =23, 
         ckpt: str = None, base: str = None, mode: int = 2, rate: int = 10000, g_mode='+', device=0, input_prompt='a photo'):
    """
    Prepare the config and the model used for generating adversarial examples.
    :param epsilon: Strength of adversarial attack in l_{\infinity}.
                    After the round and the clip process during adversarial attack, 
                    the final perturbation budget will be (epsilon+1)/255.
    :param steps: Iterations of the attack.
    :param alpha: strength of the attack for each step. Measured in l_{\infinity}.
    :param input_size: Size of the input image.
    :param object: Set True if the targeted images describes a specifc object instead of a style.
    :param mode: The mode for computation of the loss. 0: semantic; 1: textural; 2: fused. 
                 See the document for more details about the mode.
    :param rate: The fusion weight. Higher rate refers to more emphasis on semantic loss.
    :returns: a dictionary containing model and config.
    """

    if ckpt is None:
        ckpt = 'models/sd-v1-2.ckpt'

    if base is None:
        base = 'configs/stable-diffusion/v1-inference-attack.yaml'

    # seed_everything(seed)
    imagenet_templates_small_style = ['a painting']
    
    imagenet_templates_small_object = ['a photo']

    config_path = os.path.join(os.getcwd(), base)
    config = OmegaConf.load(config_path)

    ckpt_path = os.path.join(os.getcwd(), ckpt)
    model = load_model_from_config(config, ckpt_path, device=device).to(device)

    fn = identity_loss()

    # if object:
    #     imagenet_templates_small = imagenet_templates_small_object
    # else:
    #     imagenet_templates_small = imagenet_templates_small_style

    # input_prompt = [imagenet_templates_small[0] for i in range(1)]
    
    net = target_model(model, input_prompt, mode=mode, rate=rate, g_mode=g_mode)
    net.eval()

    # parameter
    parameters = {
        'epsilon': epsilon / 255.0 * (1 - (-1)),
        'alpha': alpha / 255.0 * (1 - (-1)),
        'steps': steps,
        'input_size': input_size,
        'mode': mode,
        'rate': rate,
        'g_mode': g_mode
    }

    return {'net': net, 'fn': fn, 'parameters': parameters}


def infer(img: PIL.Image.Image, mask_img: PIL.Image.Image, config, tar_img: PIL.Image.Image, 
          diff_pgd=None, using_target=False, device=0) -> np.ndarray:
    """
    Process the input image and generate the misted image.
    :param img: The input image or the image block to be misted.
    :param config: config for the attack.
    :param img: The target image or the target block as the reference for the textural loss.
    :returns: A misted image.
    """

    net = config['net']
    fn = config['fn']
    parameters = config['parameters']
    mode = parameters['mode']
    epsilon = parameters["epsilon"]
    g_mode = parameters['g_mode']
    
    cprint(f'epsilon: {epsilon}', 'y')
    
    alpha = parameters["alpha"]
    steps = parameters["steps"]
    input_size = parameters["input_size"]
    rate = parameters["rate"]
    
    
    img = img.convert('RGB')
    img = np.array(img).astype(np.float32) / 127.5 - 1.0
    img = img[:, :, :3]

    tar_img = np.array(tar_img).astype(np.float32) / 127.5 - 1.0
    tar_img = tar_img[:, :, :3]
    
    data_source = torch.zeros([1, 3, input_size, input_size]).to(device)
    data_source[0] = totensor(img).to(device)

    target_info = torch.zeros([1, 3, input_size, input_size]).to(device)
    target_info[0] = totensor(tar_img).to(device)

    net.target_info = target_info
    if mode == 'texture_self_enhance':
        net.target_info = data_source

    net.mode = mode
    net.rate = rate
    label = torch.zeros(data_source.shape).to(device)

    # Targeted PGD attack is applied.
    
    time_start_attack = time.time()

    if mode in ['advdm', 'texture_only', 'mist', 'texture_self_enhance']: # using raw PGD
            
        attack = LinfPGDAttack(net, fn, epsilon, steps, eps_iter=alpha, clip_min=-1.0, targeted=True)
        attack_output = attack.perturb(data_source, label)
    
    elif mode == 'sds': # apply SDS to speed up the PGD
        
        print('using sds')
        
        attack =  Linf_PGD(net, fn, epsilon, steps=steps, eps_iter=alpha, clip_min=-1.0, targeted=True, attack_type='PGD_SDS', g_mode=g_mode)
        
        attack_output, loss_all = attack.pgd_sds(X=data_source, net=net, c=net.condition,
                                                diff_pgd=diff_pgd, using_target=using_target, target_image=target_info, target_rate=rate)
        
        print(torch.abs(attack_output-data_source).max())
        # print(loss_all)
        # emp = [wandb.log({'adv_loss':loss_item}) for loss_item in loss_all]
    
    elif mode == 'sds_z':
        print('using sds')
        
        attack =  Linf_PGD(net, fn, epsilon, steps=steps, eps_iter=alpha, clip_min=-1.0, targeted=True, attack_type='PGD_SDS', g_mode=g_mode)
        dm = net.model
        with torch.no_grad():
            z = dm.get_first_stage_encoding(dm.encode_first_stage(data_source)).to(device)

        attack_output, loss_all = attack.pgd_sds_latent(z=z, net=net, c=net.condition,
                                                diff_pgd=diff_pgd, using_target=using_target, target_image=target_info, target_rate=rate)
        
        
    elif mode == 'none':
        attack_output = data_source
    
    print('Attack takes: ', time.time() - time_start_attack)

    editor = SDEdit(net=net)
    edit_one_step = editor.edit_list(attack_output, restep=None, t_list=[0.01, 0.05, 0.1, 0.2, 0.3])
    edit_multi_step  = editor.edit_list(attack_output, restep='ddim100')
    

    pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        safety_checker=None,
    )
    pipe_inpaint = pipe_inpaint.to("cuda")

    prompt = 'a man in a restaurant'

    mask_img = mask_img.convert('L')
    mask_img = ImageOps.invert(mask_img)
    mask_img = totensor(mask_img).to("cuda")
        
    strength = 1
    guidance_scale = 7.5
    num_inference_steps = 100
    attack_output = torch.clamp((attack_output + 1.0) / 2.0, min=0.0, max=1.0).detach()

    inpaint = pipe_inpaint(
                prompt=prompt, 
                image=attack_output, 
                mask_image=mask_img, 
                eta=1,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength
    ).images[0]


    attack_output = attack_output.cpu()[0]
    mask_img = mask_img.cpu()
    inpaint = recover_image(inpaint, attack_output, mask_img)


    return attack_output,  edit_one_step, edit_multi_step, inpaint




@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "./configs/attack"), config_name="base")
def main(cfg : DictConfig):
    print(cfg.attack)
    time_start = time.time()
    
    args = cfg.attack
    
    image_dir, mask_dir = args.image_dir, args.mask_dir
    output_path, target_image_path = args.output_path, args.target_image_path
    
    epsilon = args.epsilon
    steps = args.steps
    input_size = args.input_size
    mode = args.mode
    alpha = args.alpha
    g_mode = args.g_mode
    diff_pgd = args.diff_pgd
    using_target = args.using_target
    rate = args.target_rate if not mode == 'mist' else 1e4

    device=args.device
    
    
    if using_target and mode == 'sds':
        mode_name = f'{mode}T{rate}'
    else:
        mode_name = mode

    if mode == 'none':
        output_path = output_path + f'/{mode_name}'
    else:
        output_path = output_path + f'/{mode_name}_eps{epsilon}_steps{steps}_gmode{g_mode}'
        
    if diff_pgd[0]:
        output_path = output_path + '_diffpgd/'
    else:
        output_path += '/'
    
    mp(output_path)
    
    input_prompt = 'a photo'
    if 'anime' in image_dir:
        input_prompt = 'an anime picture'
    elif 'artwork' in image_dir:
        input_prompt = 'an artwork painting'
    elif 'landscape' in image_dir:
        input_prompt = 'a landscape photo'
    elif 'portrait' in image_dir:
        input_prompt = 'a portrait photo'
    else:
        input_prompt = 'a photo'
        
    
    config = init(epsilon=epsilon, alpha=alpha, steps=steps, 
                  mode=mode, rate=rate, g_mode=g_mode, device=device, 
                  input_prompt=input_prompt)

    
    image_paths = glob.glob(image_dir + '/*.png') + glob.glob(image_dir +'/*.jpg') \
                + glob.glob(image_dir + '/*.jpeg') + glob.glob(image_dir + '/*.webp') 
    
    image_paths.sort(key=lambda x: int(x[x.rfind('/') + 1 : x.rfind('.')]))
    
    image_paths = image_paths[:args.max_exp_num]
    
    for image_path in tqdm(image_paths):
        cprint(f'Processing: [{image_path}]', 'y')
        
        file_name = image_path.rsplit('/')[-1].rsplit('.')[0]
        mp(output_path + file_name)
        mask_path = mask_dir + file_name + "_mask.png"
        
        img = load_image_from_path(image_path, input_size)
        tar_img = load_image_from_path(target_image_path, input_size)
        mask_img = load_image_from_path(mask_path, input_size)
      
        
        output, edit_one_step, edit_multi_step, inpaint = \
        infer(img, mask_img, config, tar_img, diff_pgd=diff_pgd, using_target=using_target, device=device)  
        
        si(output, output_path + f'/{file_name}_attacked.png')
        si(edit_one_step, output_path + f'{file_name}_onestep.png')
        si(edit_multi_step, output_path + f'{file_name}_multistep.png')
        si(inpaint, output_path + f'{file_name}_inpaint.png')
        
        
        print('TIME CMD=', time.time() - time_start)


if __name__ == '__main__':
    main()