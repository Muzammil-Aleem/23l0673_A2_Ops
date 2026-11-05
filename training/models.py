"""Task 2.2 (baseline CNN) and 2.3 (transfer learning) model builders.

Every model takes raw pixels in [0, 255] and does its own scaling inside the
graph, so the saved .keras file carries its preprocessing with it.
"""
import keras
from keras import layers

# Where the "last block" starts in each backbone (layer name prefix).
LAST_BLOCK_PREFIX = {"mobilenetv2": "block_16", "efficientnetb0": "block7a"}


def build_cnn(img_size: int = 224, dropout: float = 0.4) -> keras.Model:
    """Small from-scratch CNN: 4 conv stages of Conv-BN-ReLU + MaxPool, then GAP + dropout + sigmoid."""
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1 / 255.0, name="scale_0_1")(inputs)  # pixels -> [0, 1] for stable gradients
    for i, filters in enumerate([32, 64, 128, 256], start=1):
        # no conv bias: BatchNorm's beta does the same job
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)  # 256 numbers instead of a huge Flatten
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def build_transfer(backbone: str = "mobilenetv2", img_size: int = 224, dropout: float = 0.3,
                   head_units: int = 128) -> keras.Model:
    """Pretrained ImageNet backbone without its classifier + our own small head. Backbone frozen."""
    inputs = keras.Input((img_size, img_size, 3), name="image")
    if backbone == "mobilenetv2":
        # identical to keras.applications.mobilenet_v2.preprocess_input: x / 127.5 - 1 -> [-1, 1]
        x = layers.Rescaling(1 / 127.5, offset=-1.0, name="mobilenet_preprocess")(inputs)
        base = keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                              input_shape=(img_size, img_size, 3))
    elif backbone == "efficientnetb0":
        x = inputs  # EfficientNet has its Rescaling+Normalization built in and expects 0..255
        base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                                 input_shape=(img_size, img_size, 3))
    else:
        raise ValueError(f"unknown backbone {backbone}")
    base.trainable = False
    # training=False keeps BatchNorm in inference mode even after we unfreeze layers,
    # so the small dataset does not wreck the ImageNet running statistics.
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout_1")(x)
    x = layers.Dense(head_units, activation="relu", name="head_dense")(x)
    x = layers.Dropout(dropout, name="dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return keras.Model(inputs, outputs, name=f"{backbone}_transfer")


def get_backbone(model: keras.Model) -> keras.Model:
    return next(l for l in model.layers if isinstance(l, keras.Model))


def unfreeze(model: keras.Model, backbone: str, n_layers: int | None = None) -> list[str]:
    """Unfreeze the last block of the backbone (default) or its last n_layers layers.
    BatchNorm layers stay frozen. Returns the names of trainable backbone layers."""
    base = get_backbone(model)
    base.trainable = True
    if n_layers is None:
        prefix = LAST_BLOCK_PREFIX[backbone]
        start = next(i for i, l in enumerate(base.layers) if l.name.startswith(prefix))
    else:
        start = max(len(base.layers) - n_layers, 0)
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= start and not isinstance(layer, layers.BatchNormalization)
    return [l.name for l in base.layers if l.trainable and l.weights]


def trainable_layer_table(model: keras.Model) -> list[tuple[str, int]]:
    """(layer name, trainable params) for every layer — nested backbone layers included."""
    rows = []
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            for sub in layer.layers:
                if sub.trainable and sub.trainable_weights:
                    rows.append((f"{layer.name}/{sub.name}",
                                 sum(int(w.numpy().size) for w in sub.trainable_weights)))
        elif layer.trainable_weights:
            rows.append((layer.name, sum(int(w.numpy().size) for w in layer.trainable_weights)))
    return rows
