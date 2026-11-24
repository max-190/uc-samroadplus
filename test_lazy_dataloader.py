from utils import load_config
from dataset import SatMapDataset, graph_collate_fn
from torch.utils.data import DataLoader

config = load_config("config/toponet_vitb_512_globalscale.yaml")

print("Loading training dataset...", flush=True)
train_ds = SatMapDataset(config, split="train", dev_run=True)

print("Loading validation dataset...", flush=True)
val_ds = SatMapDataset(config, split="val", dev_run=True)

print("Train tiles:", len(train_ds.tile_records))
print("Validation tiles:", len(val_ds.tile_records))

print("\nTesting training DataLoader...", flush=True)
train_loader = DataLoader(
    train_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    collate_fn=graph_collate_fn,
)

batch = next(iter(train_loader))

print("Training batch:")
for key, value in batch.items():
    print(key, value.shape if hasattr(value, "shape") else type(value))

print("\nTesting validation DataLoader...", flush=True)

val_loader = DataLoader(
    val_ds,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    collate_fn=graph_collate_fn,
)

batch = next(iter(val_loader))

print("Validation batch:")
for key, value in batch.items():
    print(key, value.shape if hasattr(value, "shape") else type(value))

print("\nBoth DataLoader tests completed successfully.")