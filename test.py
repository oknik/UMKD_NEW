import os
import torch
import numpy as np
from tqdm import tqdm
import argparse

from utils.stream_metrics import StreamClsMetrics
from utils.metric import MAE, accuracy, specific_eval, predictive_entropy, mutual_information
from datasets import APTOSDataset, DRDataset, SICAPv2Dataset, INDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from models.resnet_SDD_REDL_multi import resnet18_sdd_redl, resnet34_sdd_redl, resnet50_sdd_redl
from models.resnet_SDD_LP import resnet18_sdd_lp

# ======================
# model dict
# ======================
_model_dict = {
    'resnet18': resnet18_sdd_lp,
    'resnet18_sdd_redl': resnet18_sdd_redl,
    'resnet34': resnet34_sdd_redl,
    'resnet50': resnet50_sdd_redl,
}

# ======================
# parser
# ======================
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='/root/autodl-tmp/UMKD_new/IN')
    parser.add_argument("--model", type=str, default='resnet18')
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, default='in',
                        choices=['eyepacs', 'aptos', 'sicapv2', 'aptos01234', 'in'])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu_id", type=str, default='0')
    return parser


# ======================
# dataset setup
# ======================
def get_dataset(opts):
    if opts.dataset == 'aptos':
        dataset = APTOSDataset
        num_classes = 5
    elif opts.dataset == 'aptos01234':
        dataset = APTOSDataset
        num_classes = 5
    elif opts.dataset == 'sicapv2':
        dataset = SICAPv2Dataset
        num_classes = 4
    elif opts.dataset == 'eyepacs':
        dataset = DRDataset
        num_classes = 5
    elif opts.dataset == 'in':
        dataset = INDataset
        num_classes = 3
    else:
        raise ValueError

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_data = dataset(None, None, 'test', transform, 0, False)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=opts.batch_size,
        shuffle=False
    )

    return test_loader, num_classes


# ======================
# 6-channel support
# ======================
def convert_to_6ch(model):
    old_conv = model.conv1
    model.conv1 = torch.nn.Conv2d(
        6, 64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )
    return model


# ======================
# validation
# ======================
def validate(model, loader, device, num_classes):
    model.eval()
    metrics = StreamClsMetrics(num_classes)

    total_logits = []
    total_targets = []

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs, _, _ = model(images)

            preds = outputs
            metrics.update(preds, labels)

            total_logits.append(preds)
            total_targets.append(labels)

    total_logits = torch.cat(total_logits, dim=0)
    total_targets = torch.cat(total_targets, dim=0)

    score = metrics.get_results()
    mae = MAE(total_logits, total_targets)

    entropy = predictive_entropy(total_logits)
    mi = mutual_information(total_logits, total_targets)

    dic_data, measure_result = specific_eval(total_logits, total_targets)

    f1 = dic_data["weighted avg"]["f1-score"]
    recall = dic_data["macro avg"]["recall"]
    precision = dic_data["macro avg"]["precision"]

    return score, mae, f1, recall, precision, entropy, mi, measure_result


# ======================
# main
# ======================
def main():
    opts = get_parser().parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader, num_classes = get_dataset(opts)

    print("Loading model...")
    model = _model_dict[opts.model](pretrained=False, num_classes=num_classes, M='[1,2,4]').to(device)

    if opts.dataset == 'in':
        model = convert_to_6ch(model).to(device)

    ckpt_path = os.path.join(opts.ckpt, "max_acc.pth")
    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    print("Running evaluation...")
    score, mae, f1, recall, precision, entropy, mi, measure_result = validate(
        model, test_loader, device, num_classes
    )

    save_path = os.path.join(opts.ckpt, "test_results.txt")

    with open(save_path, "w") as f:
        f.write("===== Test Results =====\n")
        f.write("Confusion Matrix:\n")
        f.write(str(score['Confusion Matrix']) + "\n\n")

        f.write(f"Overall Acc: {score['Overall Acc']}\n")
        f.write(f"Mean Acc: {score['Mean Acc']}\n")
        f.write(f"MAE: {mae}\n")
        f.write(f"F1: {f1}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Entropy: {entropy}\n")
        f.write(f"Mutual Information: {mi}\n\n")

        f.write("Detailed Metrics:\n")
        f.write(str(measure_result) + "\n")

    print("Results saved to:", save_path)


if __name__ == '__main__':
    main()