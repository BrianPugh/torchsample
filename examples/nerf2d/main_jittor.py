import argparse
from pathlib import Path
from time import time

import cv2
import jittor as jt
import jittor.transform as transforms
import numpy as np
from jittor.dataset import Dataset
from tqdm import tqdm

import torchsample.jittor as ts

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.ImageNormalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


class SingleImageDataset(Dataset):
    def __init__(self, fn, batch_size, num_workers=0):
        super().__init__(
            batch_size=batch_size, shuffle=True, num_workers=num_workers, endless=True
        )
        self.image = cv2.cvtColor(cv2.imread(str(fn)), cv2.COLOR_BGR2RGB)
        self.image = self.image.transpose(
            (2, 0, 1)
        )  # Should this be handled automatically in jittor?
        self.image = transform(self.image)  # (3, h, w)
        self.image = jt.float(self.image)
        self.batch_size = batch_size
        self.size = self.image.shape[-1], self.image.shape[-2]  # (x, y)
        self.total_len = 1

    def __getitem__(self, idx):
        out = {}
        out["coords"] = ts.coord.randint(0, self.batch_size, self.size)
        out["rgb"] = ts.sample.nobatch(out["coords"], self.image, mode="nearest")
        return out


def main():
    parser = argparse.ArgumentParser(
        description="NeRF 2D Example.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("input/polite.jpg"),
        help="Input image to learn.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16384, help="Number of samples per minibatch."
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(2e3),
        help="Number of training iterations.",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=200,
        help="Every this many training iterations, perform a full query "
        "and save the prediction.",
    )
    parser.add_argument(
        "--pos-enc", action="store_true", help="Use gamma positional encoding."
    )
    args = parser.parse_args()

    output_folder = Path("output") / f"{args.input.stem}_pos-enc={args.pos_enc}"
    output_folder.mkdir(parents=True, exist_ok=True)

    if args.pos_enc:
        mlp_in = 40
    else:
        mlp_in = 2
    model = ts.models.MLP(mlp_in, 256, 256, 256, 3)
    optimizer = jt.optim.AdamW(model.parameters(), lr=args.lr)

    dataset = SingleImageDataset(args.input, args.batch_size, num_workers=4)

    print("Begin Training")
    pbar = tqdm(zip(range(args.iterations), dataset))

    t_start = time()
    t_save = 0
    for iteration, batch in pbar:
        optimizer.zero_grad()
        coords = batch["coords"]
        if args.pos_enc:
            coords = ts.encoding.gamma(coords)
        pred = model(coords)

        loss = jt.nn.l1_loss(pred, batch["rgb"])

        pbar.set_description(f"loss: {loss:.3f}")

        optimizer.step(loss)

        if (iteration + 1) % args.save_freq == 0 or iteration == args.iterations - 1:
            t_save -= time()
            coords = ts.coord.full_like.nobatch(dataset.image)

            if args.pos_enc:
                coords = ts.encoding.gamma(coords)

            with jt.no_grad():
                raster = model(coords)
                raster = raster.numpy()

            # Undo the normalization
            raster = (raster * 0.5) + 0.5
            raster = np.clip((raster * 255).round(), 0, 255).astype(np.uint8)
            out_fn = output_folder / f"{iteration + 1}.jpg"
            cv2.imwrite(str(out_fn), cv2.cvtColor(raster, cv2.COLOR_RGB2BGR))
            t_save += time()
    t_end = time()
    t_optim = t_end - t_start - t_save
    print(f"Optimized {args.iterations} in {t_optim:.3f}s.")


if __name__ == "__main__":
    main()
