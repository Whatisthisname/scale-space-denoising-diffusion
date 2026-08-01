import math
import os
from models.UNET import UNet

import torch
import torchvision
import torch.nn as nn

from utils import create_mnist_dataloaders, pick_device


class DDPM(nn.Module):
    def __init__(self, image_size, ctx_sz=1, markov_states=1000, unet_stages=3, noise_schedule_param=2.0, device=None):
        super().__init__()
        self.markov_states = markov_states
        self.image_size = image_size
        self.device = pick_device() if device is None else torch.device(device)
        self.model = UNet(unet_stages, ctx_sz).to(self.device)

        self.register_buffer(
            "betas", _cosine_variance_schedule(markov_states, power=noise_schedule_param).to(self.device)
        )
        self.register_buffer("alphas", (1.0 - self.betas).to(self.device))
        self.register_buffer("alphas_cumprod", self.alphas.cumprod(dim=-1).to(self.device))
        self.register_buffer("sqrt_alphas_cumprod", self.alphas_cumprod.sqrt().to(self.device))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - self.alphas_cumprod).sqrt().to(self.device))

    def training_loss(self, clean_image: torch.Tensor, labels: torch.Tensor):
        """Loss on a batch of clean images: the model predicts the noise, we return the MSE. Minimize the output directly.

        Deliberately not named `train`, which would shadow nn.Module.train and make
        eval() unreachable, leaving BatchNorm in batch-statistics mode while sampling.
        """
        noise = torch.randn_like(clean_image).to(self.device)
        t = torch.randint(0, self.markov_states - 1, (clean_image.shape[0],)).to(self.device)

        noisy = self.forward_diffusion(clean_image, noise, t, keep_intermediate=False).to(self.device)

        context = self.make_context(clean_image.shape[0], t, labels).to(self.device)

        pred_noise = self.model(noisy, context)

        return torch.mean((pred_noise - noise) ** 2)

    @torch.no_grad()
    def forward_diffusion(
        self, clean_images: torch.Tensor, noise: torch.Tensor, target: torch.Tensor, keep_intermediate: bool
    ) -> torch.Tensor:
        """Take a single step forwards"""

        if keep_intermediate:
            images = [clean_images]

            for t in range(self.markov_states - 1):
                image_scale = (1 - self.betas[t]).sqrt()
                noise_scale = self.betas[t].sqrt()
                # print("device of all terms below:", image_scale.device, noise_scale.device, images[-1].device, noise.device)
                noised = image_scale * images[-1] + noise_scale * torch.randn_like(clean_images).to(self.device)
                # noised = torch.clip(noised, min=-1, max=1)
                images.append(noised)

            # concatenate each step into one image for for each sample
            return torch.cat(images, dim=2)

        else:
            image_scale = self.sqrt_alphas_cumprod.gather(0, target).reshape(clean_images.shape[0], 1, 1, 1)
            noise_scale = self.sqrt_one_minus_alphas_cumprod.gather(0, target).reshape(clean_images.shape[0], 1, 1, 1)
            noised = image_scale * clean_images + noise_scale * noise
            # noised = torch.clip(noised, min=-1, max=1)
            return noised

    @torch.no_grad()
    def sample(self, amount: int, target_label: torch.Tensor, keep_intermediate: bool) -> torch.Tensor:
        """Sample from the model."""
        # sample noise from standard normal distribution
        image = torch.randn((amount, 1, self.image_size, self.image_size)).to(self.device).float()

        # print("image:", image[0, 0, :, 8])

        images = [image] if keep_intermediate else None

        for t in reversed(range(0, self.markov_states - 1)):
            t_step = t * torch.ones(amount, dtype=int).to(self.device)
            context = self.make_context(amount, t_step, target_label)
            image: torch.Tensor = self.reverse_diffusion(image, t_step, context)
            if keep_intermediate:
                images.append(image)

        if keep_intermediate:
            # images holds the images from the noisiest to the denoised image
            images = torch.stack(images, dim=1)
            return images

        else:
            return image

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
        """Invert the closed-form forward process: x0 = (x_t - sqrt(1-abar_t) * eps) / sqrt(abar_t)."""
        shape = (x_t.shape[0], 1, 1, 1)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.gather(-1, t).reshape(shape)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(shape)
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise_pred) / sqrt_alphas_cumprod_t

    @torch.no_grad()
    def reverse_diffusion(self, x_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """One ancestral sampling step, following algorithm 2 of Ho et al. 2020.

        Samples from the true posterior q(x_{t-1} | x_t, x0), which keeps the information
        already present in x_t. Re-noising a predicted x0 from scratch instead would throw
        that away and inject far more variance than the schedule calls for.
        """
        noise_pred = self.model.forward(x_t, context=context)

        shape = (x_t.shape[0], 1, 1, 1)
        beta_t = self.betas.gather(-1, t).reshape(shape)
        alpha_t = self.alphas.gather(-1, t).reshape(shape)
        alphas_cumprod_t = self.alphas_cumprod.gather(-1, t).reshape(shape)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(shape)

        mean = (x_t - (beta_t / sqrt_one_minus_alphas_cumprod_t) * noise_pred) / alpha_t.sqrt()

        if t.min() > 0:
            alphas_cumprod_prev_t = self.alphas_cumprod.gather(-1, t - 1).reshape(shape)
            posterior_variance = beta_t * (1.0 - alphas_cumprod_prev_t) / (1.0 - alphas_cumprod_t)
            return mean + posterior_variance.sqrt() * torch.randn_like(x_t)

        return mean

    @torch.no_grad()
    def insta_predict_from_t(self, x_t: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        context = self.make_context(x_t.shape[0], t, labels)
        noise_pred = self.model.forward(x_t, context)
        return self.predict_x0(x_t, t, noise_pred)

    def make_context(self, batch_size, timesteps, labels) -> torch.Tensor:
        """Create the context tensor for the given timesteps and labels"""
        # create the context tensor
        # context is a (timesteps, batch_size, 1+10) tensor
        # where the first column is the timestep and the rest are the one-hot encoded labels
        timesteps = torch.as_tensor(timesteps, device=self.device).float()
        labels = torch.as_tensor(labels, device=self.device).long()
        one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=10).float()

        context = torch.zeros(size=(batch_size, 1 + 10), device=self.device)
        context[:, 0] = timesteps / (self.markov_states - 1)
        context[:, 1:] = one_hot_labels
        return context


def _cosine_variance_schedule(timesteps, epsilon=0.003, power=10.0):
    steps = torch.linspace(0, timesteps, steps=timesteps + 1, dtype=torch.float32)
    f_t = torch.cos(((steps / timesteps + epsilon) / (1.0 + epsilon)) * math.pi * 0.5) ** power
    # betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)
    betas = torch.clip(1.0 - f_t[1:] / f_t[:timesteps], 0.0, 0.999)

    return betas


# plot the cosine variance schedule if running this file by itself
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    power = 1.0
    img_size = 16
    n_imgs = 20
    model = DDPM(img_size, markov_states=25, noise_schedule_param=power)

    train_dataloader, test_dataloader = create_mnist_dataloaders(batch_size=n_imgs, image_size=img_size)

    data = next(iter(train_dataloader))
    input_images = data[0][:n_imgs]
    input_labels = data[1][:n_imgs]

    noise = torch.randn_like(input_images)

    images = model.forward_diffusion(input_images, noise, keep_intermediate=True, target=None)

    # save the images locally
    # create the images folder if it doesn't exist

    os.makedirs("images/schedules", exist_ok=True)

    torchvision.utils.save_image(
        images,
        "images/schedules/s.png".format("test", 0),
        nrow=n_imgs,
    )

    # n = 100
    # x_axis = range(n)
    # for power in [1, 2, 5, 10, 20, 50, 100]:
    #     plt.plot(x_axis, _cosine_variance_schedule(n, power = power), label=power)
    # plt.legend()
    # plt.show()
