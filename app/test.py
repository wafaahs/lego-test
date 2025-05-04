import torch
from PIL import Image
import torchvision.transforms as transforms
import sys

from model import Generator

def cartoonize(input_path, output_path, checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Generator()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()

    image = Image.open(input_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)[0].cpu()
    output_image = transforms.ToPILImage()(output).convert("RGB")
    output_image.save(output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cartoonize(args.input, args.output, args.checkpoint)
