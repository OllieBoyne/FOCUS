from FOCUS.toc_prediction.model import FootPredictorModel
from pathlib import Path
import cv2
import os

from FOCUS.utils import image as image_utils
from tqdm import tqdm
from FOCUS.utils import normals as normals_utils
import json
from FOCUS.utils.torch_utils import get_device

import torch
import numpy as np

TOC_UNC_VMAX = 0.05
NORM_UNC_VMAX = 45.0

def _preprocess_image(img: np.ndarray):
    """Preprocess images for model input."""

    # Resize with padding
    img = image_utils.resize_preserve_aspect(img, (480, 640))

    # Normalize
    img = img / 255.0

    # Convert to tensor
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1)

    return img

def predict_toc(imgs: np.ndarray, model: FootPredictorModel | Path, out_dir: Path, img_keys: list[str],
                device=None):
    """Make predictions using a trained model.

    imgs: RGBA images."""

    device = get_device(device)

    N = len(imgs)

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return N

        def __getitem__(self, idx):
            img = imgs[idx]
            img = _preprocess_image(img)

            return {'rgb': img, 'key': img_keys[idx]}


    if isinstance(model, Path):
        model = FootPredictorModel.load(model, device=device)

    model.eval()

    dataloader = torch.utils.data.DataLoader(Dataset(), batch_size=10, shuffle=False)

    with torch.no_grad():
        with tqdm(total=N) as progress_bar:
            progress_bar.set_description("Predicting TOCs")
            for n, batch in enumerate(dataloader):

                rgb = batch['rgb'].to(device)
                predictions = model(rgb)

                batch_size = len(rgb)
                for i in range(batch_size):

                    key = batch['key'][i]
                    out_folder = out_dir / key
                    out_folder.mkdir(exist_ok=True, parents=True)

                    input_rgb = rgb[i].cpu().detach().numpy()
                    hm = predictions['hm'][i].cpu().detach().numpy()
                    mask = hm > 0.5
                    toc_img = predictions['TOC'][i].cpu().detach().numpy()

                    norm_img = predictions['norm_rgb'][i].cpu().detach().numpy()

                    # Normal uncertainty
                    alpha = normals_utils.kappa_to_alpha(predictions['norm_unc'][i].cpu().detach().numpy())
                    alpha_rgb = np.repeat(alpha, 3, axis=0) / NORM_UNC_VMAX

                    toc_std = np.exp(predictions['TOC_unc_log_var'][i].cpu().detach().numpy()) ** .5 / TOC_UNC_VMAX

                    files = {
                        'rgb': input_rgb,
                        'mask': hm,
                        'toc': toc_img,
                        'normal': norm_img,
                        'norm_unc': alpha_rgb,
                        'toc_unc': toc_std,
                    }

                    for f, data in files.items():

                        if f not in ('rgb', 'hm'):
                            data *= mask

                        data = data.transpose(1, 2, 0)

                        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(out_folder, f + '.png'), (data * 255).astype(np.uint8))

                    aux_data = {
                        'footedness': 'LR'[predictions['footedness'][i].argmax().item()],
                        'TOC_UNC_VMAX': TOC_UNC_VMAX,
                        'NORM_UNC_VMAX': NORM_UNC_VMAX,
                    }

                    with open(os.path.join(out_folder, 'aux.json'), 'w') as f:
                        json.dump(aux_data, f)

                    progress_bar.update(1)

if __name__ == '__main__':
    src = '/Users/ollie/Library/CloudStorage/OneDrive-UniversityofCambridge/FIND2D/data/Foot3D/mono3d_v11_t=56/0035'

    filenames = sorted([os.path.splitext(f)[0] for f in os.listdir(os.path.join(src, 'rgb')) if f.endswith('.png')])
    imgs = [cv2.cvtColor(cv2.imread(os.path.join(src, 'rgb', f + '.png')), cv2.COLOR_BGR2RGB) for f in filenames]
    imgs = np.array(imgs)

    predict_toc(imgs, Path('data/toc_model/resnet50_v12_t=44/model_best.pth'),
                Path('data/dummy_data_pred'), filenames
                )