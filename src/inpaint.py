from diffusers import StableDiffusionInpaintPipeline
import torch
from utils import *
from PIL import Image, ImageOps, ImageFilter


def get_dir_name_from_config(mode, g_mode, using_target, eps=16, steps=100, target_rate=5, prefix='out_iclr'):
    if using_target and mode == 'sds':
        mode_name = f'sdsT{target_rate}'
    else:
        mode_name = mode
    dir_name = f'out/{mode_name}_eps{eps}_steps{steps}_gmode{g_mode}/'
    return dir_name


EXP_LIST = [
        ('none', '-', False, -1, 'Clean'),

    ('advdm', '+', False, -1, 'AdvDM'),
    ('mist', '+', False, -1, 'MIST' ),
    ('texture_only', '+', False, -1, 'PhotoGuard'),
   ('advdm', '-', False, -1, 'AdvDM(-)'),
    ('sds', '+', False, -1, 'SDS(+)'),
    ('sds', '-', False, -1, 'SDS(-)'),
    ('sds', '-', True, 1, 'SDST1'),
    ('sds', '-', True, 5, 'SDST5'),

]

EXP_LIST = [
    ('advdm', '+', False, -1, 'AdvDM')
]


pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe_inpaint = pipe_inpaint.to("cuda")



possible_prompts = ['a people in the forest', 'a people dancing in the beach']
  
  
i=3

prompts = {
    0: 'a man in a party',
    1: 'a man near the sea',
    2: 'a woman in a wedding, with flowers in hand',
    3: 'a man in a car',
    4: 'a man on a boat',
    5: 'a man on a boat',
    6: 'a woman is eating breakfast',
    7: 'a peope on the plane, with a lot of people behind', #[feifei, good]
    8: 'a man in a party', #[CR7, good]
    9: 'a woman is eating breakfast',
    10: 'a man in a party',
    11: 'a man in a party',
    17: 'a man in a party', #[lecun good],
    19: 'a match poster of american football game', #[footbal player, not good]
    40: 'a man in wedding', #[trevor, good]
    71: 'a man dancing in a crowded party'
}

for i in range(1):
    for exp_config in EXP_LIST:
        cprint(exp_config, 'y')
        mode, g_mode, using_target, target_rate, name = exp_config
        
        dir_name = get_dir_name_from_config(mode, g_mode, using_target, target_rate=target_rate)
        cprint('fetching dir: ' + dir_name, 'g')
        
        dir_name += 'dataset/'
        
        # if i in prompts.keys():
        #     prompt = prompts[i]
        # else:
        prompt = 'a man dancing in a crowded party'
        
        init_image = Image.open('../images/elonmusk.webp').convert('RGB').resize((512,512))
        mask = load_png(f'../images/mask.png', 512)[None, ...]
        print(mask.shape)
        mask_image = get_bkg(mask)[0]
        si(mask_image, f'mask_bw.jpg')
        mask_image = Image.open(f'mask_bw.jpg').convert('RGB').resize((512,512))
        mask_image = ImageOps.invert(mask_image).resize((512,512))
        mask_image = mask_image.filter(ImageFilter.ModeFilter(size=13))
        
        # mask_image.save(f'demo_mask.jpg')
        # SEED=102
        # torch.manual_seed(SEED)
        # print(SEED)

        strength = 1
        guidance_scale = 7.5
        num_inference_steps = 100

        image_nat = pipe_inpaint(prompt=prompt, 
                            image=init_image, 
                            mask_image=mask_image, 
                            #  eta=1,
                            num_inference_steps=num_inference_steps,
                            # guidance_scale = 7.5,
                            #  strength=strength
                            ).images[0]


        image_nat.save(f'demo_origin.jpg')
        image_nat = recover_image(image_nat, init_image, mask_image)
        image_nat.save(f'demo_recovered.jpg')

