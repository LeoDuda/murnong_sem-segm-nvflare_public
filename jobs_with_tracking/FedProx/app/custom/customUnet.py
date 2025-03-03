import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D
from scripts.configuration import init_temp_conf


config = init_temp_conf()
NUM_CLASSES = len(config['data']['masks']['labels']) + 1


cfg = config["train"]
gamma = cfg["loss"]["gamma"]
loss_name = cfg["loss"]["name"]
loss_function = getattr(tfa.losses, loss_name)
alpha = cfg["loss"]["alpha"]


class CustomUnet(tf.keras.Model):
    def _init_(self, cfg):
        super(CustomUnet, self)._init_()

        # Set the random seed
        tf.random.set_seed(cfg["seed"])

        self.learning_rate = cfg["optimizer"]["lr"]
        self.alpha = cfg["loss"]["alpha"]
        self.gamma = cfg["loss"]["gamma"]
        self.metrics = [eval(v) for v in cfg["eval"]["SM_METRICS"].values()]

        self.size_h = cfg["data"]["loader"]["SIZE_H"]
        self.size_w = cfg["data"]["loader"]["SIZE_W"]

        # Check and adapt based on channel count (N)
        if cfg["data"]["loader"]["channels"] == 3:
            self.model = keras.applications.Unet(
                backbone_name=cfg["model"]["backbone"],
                encoder_weights="imagenet",
                classes=NUM_CLASSES,
                input_shape=(self.size_h, self.size_w, 3),
            )
        else:
            print(
                f"Channel count of {cfg['data']['loader']['channels']} != 3. "
                f"Adapting UNet by including a fitting first layer..."
            )

            base_model = keras.applications.Unet(
                backbone_name=cfg["model"]["backbone"],
                encoder_weights="imagenet",
                classes=NUM_CLASSES,
            )

            self.input_layer = Input(shape=(self.size_h, self.size_w, cfg["data"]["loader"]["channels"]))
            self.first_conv = Conv2D(3, (1, 1))(self.input_layer)  # map N channels to 3
            self.unet_output = base_model(self.first_conv)

            self.model = keras.models.Model(inputs=self.input_layer, outputs=self.unet_output)

        # Compile the model
        self.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss_function(alpha=alpha, gamma=gamma),
            metrics=self.metrics,
        )