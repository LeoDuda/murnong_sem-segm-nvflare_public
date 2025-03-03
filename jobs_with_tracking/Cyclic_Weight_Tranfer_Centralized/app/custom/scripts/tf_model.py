from tensorflow.keras import Model, layers
import segmentation_models as sm
from scripts.configuration import init_temp_conf


class UNet(Model):

    def __init__(
        self,
        config,
        NUM_CLASSES,
        SIZE_H,
        SIZE_W: int,
        N: int,
        loss_function,
        optimizer,
        learning_rate,
        alpha,
        gamma,
        metrics,
    ):
        super(UNet, self).__init__()

        if N == 3:
            self.model = self.create_unet(
                config["model"]["backbone"],
                NUM_CLASSES,
                SIZE_H,
                SIZE_W,
                N,
            )
        else:
            self.model = self.adapt_unet(
                config["model"]["backbone"],
                NUM_CLASSES,
                SIZE_H,
                SIZE_W,
                N,
            )

        self.model.compile(
            optimizer=optimizer(learning_rate=learning_rate),
            loss=loss_function(alpha=alpha, gamma=gamma),
            metrics=metrics,
        )
        self.build((None, SIZE_H, SIZE_W, N))

    def create_unet(self, backbone, num_classes, size_h, size_w, n):
        return sm.Unet(
            backbone,
            encoder_weights="imagenet",
            classes=num_classes,
            input_shape=(size_h, size_w, n),
        )

    def adapt_unet(self, backbone, num_classes, size_h, size_w, n):
        base_model = sm.Unet(
            backbone_name=backbone,
            encoder_weights="imagenet",
            classes=num_classes,
        )

        inp = layers.Input(shape=(size_h, size_w, n))
        layer_1 = layers.Conv2D(3, (1, 1))(
            inp
        )  # map N channels data to 3 channels
        out = base_model(layer_1)

        return Model(inputs=inp, outputs=out, name=base_model.name)

    def call(self, x):
        return self.model(x)


if __name__ == "__main__":
    # import module dependencies
    from scripts.configuration import (
        init_temp_conf,
        update_conf,
        cp_conf,
        _default_config,
    )
    import tensorflow as tf
    import tensorflow_addons as tfa

    config = init_temp_conf()
    SIZE_H = config["data"]["loader"]["SIZE_H"]
    SIZE_W = config["data"]["loader"]["SIZE_W"]
    N = 3
    # Build the model before compiling it

    cfg = config["train"]

    # (4) loads model from NVFlare
    loss_name = cfg["loss"]["name"]
    try:
        loss_function = getattr(tfa.losses, loss_name)
    except AttributeError:
        raise ValueError(
            f"Loss function '{loss_name}' not found in tensorflow_addons.losses!"
        )

    optimizer_name = cfg["optimizer"]["name"]
    try:
        optimizer = getattr(tf.keras.optimizers, optimizer_name)
    except AttributeError:
        raise ValueError(
            f"Optimizer {optimizer_name} not found in tensorflow.keras.optimizers!"
        )
    NUM_CLASSES = len(config["data"]["masks"]["labels"]) + 1
    learning_rate = cfg["optimizer"]["lr"]
    alpha = cfg["loss"]["alpha"]
    gamma = cfg["loss"]["gamma"]
    metrics = [eval(v) for v in config["eval"]["SM_METRICS"].values()]
    model = UNet(
        config,
        NUM_CLASSES,
        SIZE_H,
        SIZE_W,
        N,
        loss_function,
        optimizer,
        learning_rate,
        alpha,
        gamma,
        metrics,
    )

    model.summary()
