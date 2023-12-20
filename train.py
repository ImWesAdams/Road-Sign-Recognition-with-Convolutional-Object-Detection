import torch

from model import Detector, save_model, load_model
from utils import load_data
import torch.utils.tensorboard as tb

import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_IMAGE_PATH = 'data/train/images'
TRAIN_ANNOTATION_PATH = 'data/train/annotations'
TEST_IMAGE_PATH = 'data/test/images'
TEST_ANNOTATION_PATH = 'data/test/annotations'
SHAPE_X = 304
SHAPE_Y = 304

TRANSFORM = transforms.Compose(
    [
    transforms.ColorJitter(0.5, 0.8, 0.8, 0.5),
    transforms.Resize((SHAPE_X, SHAPE_Y)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.3, 0.3, 0.3], [0.2, 0.2, 0.2]),
    transforms.ToHeatmap((SHAPE_X, SHAPE_Y))
    ]
)

BASE_TRANSFORM = transforms.Compose(
    [
    transforms.Resize((SHAPE_X, SHAPE_Y)),
    transforms.ToTensor(),
    transforms.Normalize([0.3, 0.3, 0.3], [0.2, 0.2, 0.2]),
    transforms.ToHeatmap((SHAPE_X, SHAPE_Y))
    ]
)


def train(args):
    from os import path
    if args.model == 'none':
        model = Detector(output_channels = 5)
    else:
        model = load_model(args.model, output_channels = 5)
    model = model.to(device)
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
    optim = torch.optim.SGD(model.parameters(), lr = float(args.learning_rate), momentum = float(args.nmomentum))

    # Import data
    train_data = load_data(
        TRAIN_IMAGE_PATH,
        TRAIN_ANNOTATION_PATH,
        TRANSFORM
    )

    # Model to optimize
    optim = torch.optim.SGD(model.parameters(), lr = float(args.learning_rate), momentum = float(args.nmomentum))

    # Loss calculation method
    loss_calc = torch.nn.BCEWithLogitsLoss(reduction='none')

    # Define a learning rate step function on loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience = 800, factor = 0.5)

    global_step = int(args.global_step)

    # Iterate
    for i in range(int(args.iterations)):
        model.train()
        for x_val, y_val, image in train_data:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            # Make a prediction and calculate loss
            pred = model.forward(x_val).to(device)
            # print(x_val.shape, y_val.shape, pred.shape)
            loss = loss_calc(pred, y_val).to(device)
            loss = loss.mean()
            # Reset the gradient
            optim.zero_grad()
            # Pass the loss backwards, balancing mse loss by factor of 1/200
            loss.backward()
            # Step
            optim.step()
            accuracy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # # Accuracy
            # accuracy = calculate_accuracy(y_val, pred)
            learning_rate = scheduler.optimizer.param_groups[0]['lr']
            # Log training images, predictions, loss, and learning rate
            log(train_logger, x_val, y_val, image, pred, loss, accuracy, learning_rate, global_step)
            # Move global step up
            print('train:', global_step, loss)
            # Step the scheduler
            scheduler.step(loss)
            global_step += 1
        # Save model and test after each full loop through training data
        model.eval()
        save_model(model, str(global_step))
        test(global_step, str(global_step))

def test(global_step, model):
    with torch.no_grad():
        from os import path
        model = load_model(model + '_det.th', output_channels = 5)
        model = model.to(device)
        model.eval()
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

        # Import data
        test_data = load_data(
        TEST_IMAGE_PATH,
        TEST_ANNOTATION_PATH,
        BASE_TRANSFORM
        )

        # Loss calculation method
        loss_calc = torch.nn.BCEWithLogitsLoss(reduction='none')

        total_loss = 0
        total_accuracy = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(1):
            for x_val, y_val, image in test_data:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                # Make a prediction and calculate loss and accuracy
                pred = model.forward(x_val).to(device)
                # print(x_val.shape, y_val.shape, pred.shape)
                loss = loss_calc(pred, y_val).to(device) # peak
                loss = loss.mean()
                total_loss += loss
                accuracy = calculate_accuracy(y_val, pred)
                for idx in range(len(total_accuracy)):
                    total_accuracy[idx] = total_accuracy[idx] + accuracy[idx]
                print('test:', global_step, loss)
        model.eval()
        # Log training images, predictions, loss, and learning rate
        log(valid_logger, x_val, y_val, image, pred, total_loss, total_accuracy, 0, global_step)

def log(logger, x_val, y_val, image, pred, loss, accuracy, learning_rate, global_step):
    # Add transformed images
    logger.add_images('x_val', x_val[:4], global_step)
    # Pull in, stack, and add label images
    y_val = y_val[:4]
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
    logger.add_images('0y_tl', y_tl,  global_step)
    logger.add_images('1y_s', y_s,  global_step)
    logger.add_images('2y_sl', y_sl,  global_step)
    logger.add_images('3y_cw', y_cw,  global_step)
    logger.add_images('4y_e', y_e,  global_step)
    logger.add_images('image', image[0][:4], global_step)
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
    logger.add_images('0pred_tl', torch.sigmoid(pred_tl), global_step)
    logger.add_images('1pred_s', torch.sigmoid(pred_s), global_step)
    logger.add_images('2pred_sl', torch.sigmoid(pred_sl), global_step)
    logger.add_images('3pred_cw', torch.sigmoid(pred_cw), global_step)
    logger.add_images('4pred_e', torch.sigmoid(pred_e), global_step)
    # Add loss and accuracy scalars
    logger.add_scalar('loss', loss, global_step)
    logger.add_scalar('acc_p', accuracy[0], global_step)
    logger.add_scalar('0acc_tl_p', accuracy[2], global_step)
    logger.add_scalar('1acc_s_p', accuracy[4], global_step)
    logger.add_scalar('2acc_sl_p', accuracy[6], global_step)
    logger.add_scalar('3acc_cw_p', accuracy[8], global_step)
    logger.add_scalar('4acc_e_p', accuracy[10], global_step)
    logger.add_scalar('acc_n', accuracy[1], global_step)
    logger.add_scalar('0acc_tl_n', accuracy[3], global_step)
    logger.add_scalar('1acc_s_n', accuracy[5], global_step)
    logger.add_scalar('2acc_sl_n', accuracy[7], global_step)
    logger.add_scalar('3acc_cw_n', accuracy[9], global_step)
    logger.add_scalar('4acc_e_n', accuracy[11], global_step)
    # Log learning rate
    logger.add_scalar('learning_rate', learning_rate, global_step)

def calculate_accuracy(y_val, pred):
    # Log positive accuracies (points in boxes) and negative accuracies (points not in boxes)
    pred = torch.sigmoid(pred)
    positive = y_val > 0
    acc_p = y_val[positive] - pred[positive]
    negative = y_val == 0
    acc_n = y_val[negative] - pred[negative]
    accuracies = [acc_p.sum(), acc_n.sum()]
    for i in range(5):
        y = y_val[:, i, :, :]
        p = pred[:, i, :, :]
        y_p = y > 0
        y_n = y == 0
        acc_p = y[y_p] - p[y_p]
        acc_n = y[y_n] - p[y_n]
        accuracies.append(acc_p.sum())
        accuracies.append(acc_n.sum())
    return accuracies


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default='runs')
    parser.add_argument('-n', '--nmomentum', default = 0.9)
    parser.add_argument('-l', '--learning_rate', default = 0.1)
    parser.add_argument('-i', '--iterations', default = 1000)
    parser.add_argument('-c', '--continue_training', action = 'store_true')
    parser.add_argument('-g', '--global_step', default = 0)
    parser.add_argument('-v', '--validate', action = 'store_true')
    parser.add_argument('-m', '--model', default = 'none')

    args = parser.parse_args()
    if args.validate:
        test(args)
    else:
        train(args)