import torch
from torchvision.utils import save_image

from model import Detector, save_model, load_model
from utils import load_data

import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_IMAGE_PATH = 'data/test/images'
TEST_ANNOTATION_PATH = 'data/test/annotations'
SHAPE_X = 304
SHAPE_Y = 304

BASE_TRANSFORM = transforms.Compose(
    [
    transforms.Resize((SHAPE_X, SHAPE_Y)),
    transforms.ToTensor(),
    transforms.Normalize([0.3, 0.3, 0.3], [0.2, 0.2, 0.2]),
    transforms.ToHeatmap((SHAPE_X, SHAPE_Y))
    ]
)

def test(model):
    with torch.no_grad():
        model = load_model(model + '_det.th', output_channels = 5)
        model = model.to(device)
        model.eval()

        # Import data
        test_data = load_data(
        TEST_IMAGE_PATH,
        TEST_ANNOTATION_PATH,
        BASE_TRANSFORM
        )

        i = 0
        total_positive_pixels = 0
        total_negative_pixels = 0
        traffic_light_positive = 0
        speed_limit_positive = 0
        stop_sign_positive = 0
        crosswalk_sign_positive = 0

        for x_val, y_val, image in test_data:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            positive = y_val[y_val > 0]
            positive = positive.sum()
            negative = y_val[y_val == 0]
            negative = negative.nelement()
            traffic_light_positive += y_val[:, 0, :, :].sum()
            speed_limit_positive += y_val[:, 1, :, :].sum()
            stop_sign_positive += y_val[:, 2, :, :].sum()
            crosswalk_sign_positive += y_val[:, 3, :, :].sum()

            # Make a prediction
            pred = model.forward(x_val).to(device)

            store_images(i, y_val, image[0], pred)
            total_positive_pixels += positive
            total_negative_pixels += negative
            i += 1
    print('Total positive pixels:', total_positive_pixels)
    print('Total negative pixels:', total_negative_pixels)
    print('traffic light positive:', traffic_light_positive)
    print('speed limit positive:', speed_limit_positive)
    print('stop sign positive:', stop_sign_positive)
    print('crosswalk sign positive:', crosswalk_sign_positive)

def store_images(iter, y_val, image, pred):
    # Pull in, stack, and add label images
    y_tl = y_val[:, 0, :, :]
    y_s = y_val[:, 1, :, :]
    y_sl = y_val[:, 2, :, :]
    y_cw = y_val[:, 3, :, :]
    y_e = y_val[:, 4, :, :]
    y_tl = torch.stack([y_tl, y_tl, y_tl], dim = 1)
    y_s = torch.stack([y_s, y_s, y_s], dim = 1)
    y_sl = torch.stack([y_sl, y_sl, y_sl], dim = 1)
    y_cw = torch.stack([y_cw, y_cw, y_cw], dim = 1)
    y_e = torch.stack([y_e, y_e, y_e], dim = 1)
    # Pull in, stack, and add prediction images
    pred = pred[:4]
    pred_tl = pred[:, 0, :, :]
    pred_s = pred[:, 1, :, :]
    pred_sl = pred[:, 2, :, :]
    pred_cw = pred[:, 3, :, :]
    pred_e = pred[:, 4, :, :]
    pred_tl = torch.stack([pred_tl, pred_tl, pred_tl], dim = 1)
    pred_s = torch.stack([pred_s, pred_s, pred_s], dim = 1)
    pred_sl = torch.stack([pred_sl, pred_sl, pred_sl], dim = 1)
    pred_cw = torch.stack([pred_cw, pred_cw, pred_cw], dim = 1)
    pred_e = torch.stack([pred_e, pred_e, pred_e], dim = 1)
    for i in range(4):
        save_image(y_tl[i], str(iter) + '_' + str(i) + 'y_tl.png')
        save_image(y_s[i], str(iter) + '_' + str(i) + 'y_s.png')
        save_image(y_sl[i], str(iter) + '_' + str(i) + 'y_sl.png')
        save_image(y_cw[i], str(iter) + '_' + str(i) + 'y_cw.png')
        save_image(pred_tl[i], str(iter) + '_' + str(i) + 'pred_tl.png')
        save_image(pred_s[i], str(iter) + '_' + str(i) + 'pred_s.png')
        save_image(pred_sl[i], str(iter) + '_' + str(i) + 'pred_sl.png')
        save_image(pred_cw[i], str(iter) + '_' + str(i) + 'pred_cw.png')
        save_image(image[i], str(iter) + '_' + str(i) + 'image.png')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default = 'none')

    args = parser.parse_args()
    test(args.model)